"""Tests for Chamfer distance computation."""

import numpy as np

from src.evaluation.chamfer import (
    extract_edge_points,
    chamfer_distance,
    chamfer_score,
)


def test_extract_edge_points():
    img = np.zeros((100, 100), dtype=np.uint8)
    img[50, 30:70] = 255  # horizontal line
    points = extract_edge_points(img)
    assert points.shape[1] == 2
    assert len(points) == 40


def test_extract_edge_points_empty():
    img = np.zeros((100, 100), dtype=np.uint8)
    points = extract_edge_points(img)
    assert points.shape == (0, 2)


def test_chamfer_distance_identical():
    points = np.array([[0, 0], [1, 0], [1, 1], [0, 1]], dtype=np.float64)
    dist = chamfer_distance(points, points)
    assert dist == 0.0


def test_chamfer_distance_shifted():
    a = np.array([[0, 0], [1, 0]], dtype=np.float64)
    b = np.array([[0, 1], [1, 1]], dtype=np.float64)
    dist = chamfer_distance(a, b)
    assert abs(dist - 1.0) < 1e-6


def test_chamfer_distance_empty():
    a = np.array([[0, 0]], dtype=np.float64)
    b = np.empty((0, 2))
    dist = chamfer_distance(a, b)
    assert dist == float("inf")


def test_chamfer_score_perfect():
    img = np.zeros((100, 100), dtype=np.uint8)
    img[50, 30:70] = 255
    score = chamfer_score(img, img)
    assert score == 1.0


def test_chamfer_score_empty():
    a = np.zeros((100, 100), dtype=np.uint8)
    a[50, 30:70] = 255
    b = np.zeros((100, 100), dtype=np.uint8)
    score = chamfer_score(a, b)
    assert score == 0.0
