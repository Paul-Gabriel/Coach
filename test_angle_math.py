"""Minimal tests for angle math that don't require OpenCV or MediaPipe."""
from pose_utils import calculate_angle


def almost_equal(a: float, b: float, eps: float = 1e-4) -> bool:
    return abs(a - b) <= eps


def test_right_angle():
    # A(0,0), B(0,1), C(1,1) -> 90 deg at B
    assert almost_equal(calculate_angle((0, 0), (0, 1), (1, 1)), 90.0)


def test_straight_line_180():
    # A(0,0), B(1,0), C(2,0) -> 180 deg at B
    assert almost_equal(calculate_angle((0, 0), (1, 0), (2, 0)), 180.0)


def test_zero_angle():
    # Points make identical vectors BA and BC -> 0 deg
    # A and C are on the same ray from B
    assert almost_equal(calculate_angle((2, 2), (1, 1), (3, 3)), 0.0)


if __name__ == "__main__":
    # Simple runner
    test_right_angle()
    test_straight_line_180()
    test_zero_angle()
    print("Angle math tests: PASS")
