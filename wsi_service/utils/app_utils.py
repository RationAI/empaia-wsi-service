import re
import zipfile
from io import BytesIO

import numpy as np
import tifffile
from fastapi import HTTPException
from PIL import Image
from starlette.responses import Response

from wsi_service.models.v3.slide import SlideInfo
from wsi_service.singletons import settings
from wsi_service.utils.image_utils import (
    convert_narray_to_pil_image,
    convert_rgb_image_for_channels,
    get_requested_channels_as_array,
    get_requested_channels_as_rgb_array,
    save_rgb_image,
    get_extended_region,
)

supported_image_formats = {
    "bmp": "image/bmp",
    "gif": "image/gif",
    "jpeg": "image/jpeg",
    "png": "image/png",
    "tiff": "image/tiff",
}

alternative_spellings = {"jpg": "jpeg", "tif": "tiff"}


def process_image_region(slide, image_region, image_channels):
    if isinstance(image_region, Image.Image):
        # pillow image
        if image_channels is None:
            return image_region
        else:
            return convert_rgb_image_for_channels(image_region, image_channels)
    elif isinstance(image_region, (np.ndarray, np.generic)):
        # numpy array
        if image_channels is None:
            # workaround for now: we return first three channels as rgb
            result = get_requested_channels_as_rgb_array(image_region, None, slide)
            rgb_image = convert_narray_to_pil_image(result)
            return rgb_image
        else:
            result = get_requested_channels_as_rgb_array(image_region, image_channels, slide)
            mode = "L" if len(image_channels) == 1 else "RGB"
            rgb_image = convert_narray_to_pil_image(result, np.min(result), np.max(result), mode=mode)
            return rgb_image
    else:
        raise HTTPException(status_code=400, detail="Failed to read region in an appropriate internal representation.")


def process_image_region_raw(image_region, image_channels):
    if isinstance(image_region, Image.Image):
        # pillow image
        narray = np.asarray(image_region)
        narray = np.ascontiguousarray(narray.transpose(2, 0, 1))
        return narray
    elif isinstance(image_region, (np.ndarray, np.generic)):
        # numpy array
        if image_channels is None:
            return image_region
        else:
            result = get_requested_channels_as_array(image_region, image_channels)
            return result
    else:
        raise HTTPException(status_code=400, detail="Failed to read region in an apropriate internal representation.")


def make_response(slide, image_region, image_format, image_quality, image_channels=None):
    if isinstance(image_region, bytes):
        if image_format == "jpeg":
            return Response(image_region, media_type=supported_image_formats[image_format])
        else:
            image_region = Image.open(BytesIO(image_region))
    if image_format == "tiff":
        # return raw image region as tiff
        narray = process_image_region_raw(image_region, image_channels)
        return make_tif_response(narray, image_format)
    else:
        # return image region
        img = process_image_region(slide, image_region, image_channels)
        return make_image_response(img, image_format, image_quality)


def make_image_response(pil_image, image_format, image_quality):
    if image_format in alternative_spellings:
        image_format = alternative_spellings[image_format]

    if image_format not in supported_image_formats:
        raise HTTPException(status_code=400, detail="Provided image format parameter not supported")

    mem = save_rgb_image(pil_image, image_format, image_quality)
    return Response(mem.getvalue(), media_type=supported_image_formats[image_format])


def make_tif_response(narray, image_format):
    if image_format in alternative_spellings:
        image_format = alternative_spellings[image_format]

    if image_format not in supported_image_formats:
        raise HTTPException(status_code=400, detail="Provided image format parameter not supported for OME tiff")

    mem = BytesIO()
    try:
        if narray.shape[0] == 1:
            tifffile.imwrite(mem, narray, photometric="minisblack", compression="DEFLATE")
        else:
            tifffile.imwrite(mem, narray, photometric="minisblack", planarconfig="separate", compression="DEFLATE")
    except Exception as ex:
        raise HTTPException(status_code=400, detail=f"Error writing tiff file: {ex}")
    mem.seek(0)

    return Response(mem.getvalue(), media_type=supported_image_formats[image_format])


def validate_image_request(image_format, image_quality):
    if image_format not in supported_image_formats and image_format not in alternative_spellings:
        raise HTTPException(status_code=400, detail="Provided image format parameter not supported")
    if image_quality < 0 or image_quality > 100:
        raise HTTPException(status_code=400, detail="Provided image quality parameter not supported")


def validate_hex_color_string(padding_color):
    if padding_color:
        match = re.search(r"^#(?:[0-9a-fA-F]{3}){1,2}$", padding_color)
        if match:
            stripped_padding_color = padding_color.lstrip("#")
            int_padding_color = tuple(int(stripped_padding_color[i : i + 2], 16) for i in (0, 2, 4))
            return int_padding_color
    return settings.padding_color


def validate_image_channels(slide_info, image_channels):
    if image_channels is None:
        return
    for i in image_channels:
        if i >= len(slide_info.channels):
            raise HTTPException(
                status_code=400,
                detail=f"""
                Selected image channel exceeds channel bounds
                (selected: {i} max: {len(slide_info.channels)-1})
                """,
            )
    if len(image_channels) != len(set(image_channels)):
        raise HTTPException(status_code=400, detail="No duplicates allowed in channels")


def validate_image_size(size_x, size_y):
    if size_x * size_y > settings.max_returned_region_size:
        raise HTTPException(
            status_code=422,
            detail=f"Requested region may not contain more than {settings.max_returned_region_size} pixels.",
        )


def validate_image_z(slide_info, z):
    if z > 0 and (slide_info.extent.z == 1 or slide_info.extent.z is None):
        raise HTTPException(
            status_code=422,
            detail=f"Invalid ZStackQuery z={z}. The image does not support multiple z-layers.",
        )
    if z > 0 and z >= slide_info.extent.z:
        raise HTTPException(
            status_code=422,
            detail=f"Invalid ZStackQuery z={z}. The image has only {slide_info.extent.z} z-layers.",
        )


def validate_image_level(slide_info, level):
    if level >= len(slide_info.levels):
        raise HTTPException(
            status_code=422,
            detail="The requested pyramid level is not available. "
            + f"The coarsest available level is {len(slide_info.levels) - 1}.",
        )


async def safe_get_slide(slide_manager, path, plugin):
    try:
        return await slide_manager.get_slide(path, plugin=plugin)
    except Exception as e:
        return None  # todo consider keeping the error message


async def safe_get_slide_for_query(slide_manager, path, plugin):
    try:
        return await slide_manager.get_slide_info(path, slide_info_model=SlideInfo, plugin=plugin)
    except Exception as e:
        return {'detail': getattr(e, 'message', repr(e))}


async def safe_get_slide_info(slide):
    if slide is None:
        return None
    try:
        return await slide.get_info()
    except Exception as e:
        return None  # todo consider keeping the error message


def batch_safe_make_response(slides, image_regions, image_format, image_quality, image_channels=None):
    # Create a ZipFile and add the NumPy array as an entry named 't1'
    zip_buffer = BytesIO()
    with zipfile.ZipFile(zip_buffer, 'w', compression=zipfile.ZIP_STORED) as zip:
        for i in range(len(slides)):
            image_region = image_regions[i]
            slide = slides[i]

            if isinstance(image_region, bytes):
                if image_format == "jpeg":
                    zip.writestr(f't{i + 1}.jpeg', image_region)
                    continue
                else:
                    image_region = Image.open(BytesIO(image_region))

            if image_format == "tiff":
                mem = BytesIO()
                try:
                    # return raw image region as tiff
                    narray = process_image_region_raw(image_region, image_channels)
                    if image_format in alternative_spellings:
                        image_format = alternative_spellings[image_format]

                    if image_format not in supported_image_formats:
                        raise HTTPException(status_code=400,
                                            detail="Provided image format parameter not supported for OME tiff")
                    if narray.shape[0] == 1:
                        tifffile.imwrite(mem, narray, photometric="minisblack", compression="DEFLATE")
                    else:
                        tifffile.imwrite(mem, narray, photometric="minisblack", planarconfig="separate",
                                         compression="DEFLATE")
                    mem.seek(0)
                    print(f"Writing {i} as file raw")
                    zip.writestr(f't{i + 1}.{image_format}', mem.getvalue())
                except Exception as ex:
                    print(f"Writing {i} as err raw")
                    # just indicate error --> empty archive
                    zip.writestr(f't{i + 1}.err', getattr(ex, 'message', repr(ex)))
            else:
                try:
                    # return image region
                    img = process_image_region(slide, image_region, image_channels)
                    if image_format in alternative_spellings:
                        image_format = alternative_spellings[image_format]

                    if image_format not in supported_image_formats:
                        raise HTTPException(status_code=400, detail="Provided image format parameter not supported")

                    mem = save_rgb_image(img, image_format, image_quality)
                    print(f"Writing {i} as file")
                    zip.writestr(f't{i + 1}.{image_format}', mem.getvalue())
                except Exception as ex:
                    print(f"Writing {i} as err")
                    # just indicate error --> empty archive
                    zip.writestr(f't{i + 1}.err', getattr(ex, 'message', repr(ex)))
    return Response(zip_buffer.getvalue(), media_type="application/zip")


async def batch_safe_get_region(slide,
                                slide_info,
                                level,
                                start_x,
                                start_y,
                                size_x,
                                size_y,
                                image_channels,
                                vp_color,
                                z):
    try:
        validate_image_level(slide_info, level)
        validate_image_z(slide_info, z)
        validate_image_channels(slide_info, image_channels)
        # TODO: We don't extend tiles! No need, less data transfer, faster render
        # if check_complete_region_overlap(slide_info, level, start_x, start_y, size_x, size_y):
        #     image_region = await slide.get_region(level, start_x, start_y, size_x, size_y, padding_color=vp_color, z=z)
        # else:
        #     image_region = await get_extended_region(
        #         slide.get_region, slide_info, level, start_x, start_y, size_x, size_y, padding_color=vp_color, z=z)
        return await slide.get_region(level, start_x, start_y, size_x, size_y, padding_color=vp_color, z=z)
    except Excetion as e:
        return None


async def batch_safe_get_tile(slide,
                              slide_info,
                              level,
                              tile_x,
                              tile_y,
                              image_channels,
                              vp_color,
                              z):
    try:
        validate_image_level(slide_info, level)
        validate_image_z(slide_info, z)
        validate_image_channels(slide_info, image_channels)
        # TODO: We don't extend tiles! No need, less data transfer, faster render
        # if check_complete_tile_overlap(slide_info, level, tile_x, tile_y):
        #     image_tile = await slide.get_tile(level, tile_x, tile_y, padding_color=vp_color, z=z)
        # else:
        #     image_tile = await get_extended_tile(
        #         slide.get_tile, slide_info, level, tile_x, tile_y, padding_color=vp_color, z=z)
        tile = await slide.get_tile(level, tile_x, tile_y, padding_color=vp_color, z=z)
        return tile
    except Exception as e:
        return None
