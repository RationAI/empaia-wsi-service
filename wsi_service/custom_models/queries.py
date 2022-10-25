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
    """,
)

PluginQuery = Query(None, description="Select a specific WSI Service Plugin.")

ZStackQuery = Query(0, ge=0, description="Z-Stack layer index z")
