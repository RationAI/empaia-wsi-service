from fastapi import Query

ImageFormatsQuery = Query(
    "jpeg", description="Image format (e.g. bmp, gif, jpeg, png, tiff). For raw image data choose tiff."
)

ImageQualityQuery = Query(
    90,
    ge=0,
    le=100,
    description="""Image quality (Only for specific formats.
    For Jpeg files compression is always lossy. For tiff files 'deflate' compression is used by default.
    Set to 0 to compress lossy with 'jpeg')""",
)

ImageChannelQuery = Query(None, description="List of requested image channels. By default all channels are returned.")

ImagePaddingColorQuery = Query(
    None,
    example="#FFFFFF",
    description="""Background color as 24bit-hex-string with leading #,
    that is used when image tile contains whitespace when out of image extent. Default is white.
    Only works for 8-bit RGB slides, otherwise the background color is black.
    """,
)

PluginQuery = Query(None, description="Select a specific WSI Service Plugin.")

ZStackQuery = Query(0, ge=0, description="Z-Stack layer index z")

FileListQuery = Query(
    "",
    example="path/to/file1.ext,path/to/another/scan.ext2",
    description="""Provide file list to access simultanously via batch queries.""",
)

TileXListQuery = Query(
    None,
    example="8,6,45,0",
    description="""Provide x-coord list to access tiles at. The size must match the number of files requested.""",
)
TileYListQuery = Query(
    None,
    example="1,58,6,2",
    description="""Provide y-coord list to access tiles at. The size must match the number of files requested.""",
)
TileLevelListQuery = Query(
    None,
    example="0,5,1,2",
    description="""Provide level list to access tiles at. The size must match the number of files requested.""",
)
