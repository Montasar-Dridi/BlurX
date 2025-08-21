from insightface.app import FaceAnalysis


def insightface_face_detection(image_array, _app=[None]):
    if _app[0] is None:
        app = FaceAnalysis(
            allowed_modules=["detection"],
            providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
        )
        app.prepare(ctx_id=0, det_size=(640, 640))
        _app[0] = app

    detected_faces = app.get(image_array)
    return detected_faces
