print('====MEDIAPIPE ENVIRONMENT TESTS====')
try:
    import os
    import importlib
    import sys
    import mediapipe as mp
    import cv2
    from termcolor import colored
    import time
    print(colored('\n==========>  IMPORTS  ============>  SUCCESS!\n', 'green'))
except ImportError as e:
    print(e)


if sys.version_info >= (3, 8):
    from importlib import metadata as importlib_metadata
else:
    import importlib_metadata


def collect_distrs_list() -> [str]:
    out = []
    dists = importlib_metadata.distributions()
    for dist in dists:
        name = dist.metadata['Name']
        if not name.__contains__('-'):
            out.append(name)
    return sorted(out)


def check_import(packages_list):
    for package in packages_list:
        try:
            a = importlib.import_module(package)
            # print(a.__name__)
        except ImportError as e:
            print(e)
# print(*collect_distrs_list(), sep='\n')
# check_import(collect_distrs_list())


def check_mp_iris(image):
    mp_iris = mp.solutions.iris
    with mp_iris.Iris() as iris:
        results = iris.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    assert results is not None
    print(colored('==========>  IRIS  ===============>  SUCCESS!', 'green'))


def check_mp_facemesh(image):
    mp_face_mesh = mp.solutions.face_mesh
    with mp_face_mesh.FaceMesh(
            static_image_mode=True,
            max_num_faces=1,
            min_detection_confidence=0.1
    ) as face_mesh:
        results = face_mesh.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    assert results.multi_face_landmarks is not None
    print(colored('==========>  FACE MESH  ==========>  SUCCESS!', 'green'))


def check_mp_face_detection(image):
    mp_face_detection = mp.solutions.face_detection
    with mp_face_detection.FaceDetection(
            min_detection_confidence=0.1
    ) as face_detection:
        results = face_detection.process(image)
    assert results.detections[0] is not None
    print(colored('==========>  FACE DETECTION  =====>  SUCCESS!', 'green'))


def check_mp_holistic(image):
    mp_holistic = mp.solutions.holistic
    with mp_holistic.Holistic(
            static_image_mode=True,
            upper_body_only=True,
            min_detection_confidence=0.0
    ) as holistic:
        result = holistic.process(image)
    assert result.face_landmarks is not None
    print(colored('==========>  HOLISTIC  ===========>  SUCCESS!', 'green'))

def check():
    image = cv2.imread('test.png')
    tests = [
        check_mp_iris,
        check_mp_holistic,
        check_mp_face_detection,
        check_mp_facemesh
    ]
    for test in tests:
        test(image)


check()
