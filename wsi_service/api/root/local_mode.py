from typing import List

from fastapi.responses import JSONResponse

from wsi_service.custom_models.local_mapper_models import CaseLocalMapper, SlideLocalMapper, SlideStorage
from wsi_service.local_mapper import LocalMapper
from wsi_service.local_mapper_paths import PathsLocalMapper


def add_routes_local_mode(app, settings):
    mapper = LocalMapper if settings.local_mode_uid else PathsLocalMapper
    localmapper = mapper(settings.data_dir)

    @app.get("/cases/", response_model=List[CaseLocalMapper], tags=["Additional Routes (Standalone WSI Service)"])
    async def _():
        """
        (Only in standalone mode) Browse the local directory and return case ids for each available directory.
        """
        cases = localmapper.get_cases()
        return cases

    @app.get(
        "/cases/{case_id}/slides/",
        response_model=List[SlideLocalMapper],
        tags=["Additional Routes (Standalone WSI Service)"],
    )
    async def _(case_id: str):
        """
        (Only in standalone mode) Browse the local case directory and return slide ids for each available file.
        """
        slides = localmapper.get_slides(case_id)
        return slides

    @app.get("/slides/{slide_id}", response_model=SlideLocalMapper, tags=["Additional Routes (Standalone WSI Service)"])
    async def _(slide_id: str):
        """
        (Only in standalone mode) Return slide data for a given slide ID.
        """
        slide = localmapper.get_slide(slide_id)
        return slide

    @app.get(
        "/slides/{slide_id}/storage",
        response_model=SlideStorage,
        tags=["Additional Routes (Standalone WSI Service)"],
    )
    async def _(slide_id: str):
        """
        (Only in standalone mode) Return slide storage data for a given slide ID.
        """
        slide = localmapper.get_slide(slide_id)
        return slide.slide_storage

    @app.get("/refresh_local_mapper", tags=["Additional Routes (Standalone WSI Service)"])
    async def _():
        """
        (Only in standalone mode) Refresh available files by scanning for new files.
        """
        localmapper.refresh()
        return JSONResponse({"detail": "Local mapper has been refreshed."}, status_code=200)
