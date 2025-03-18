import os
import cv2
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio
from ultralytics import YOLO
from collections import defaultdict
import time
import datetime
import argparse
from pathlib import Path
from scipy.spatial import distance
import warnings
import threading
import queue
import socket
import json
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from PIL import Image, ImageTk
import webbrowser
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
import seaborn as sns
import matplotlib.animation as animation
from sklearn.cluster import DBSCAN
import networkx as nx
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.cm as cm
from matplotlib.colors import Normalize
import io
import base64
import requests
import re
from datetime import timedelta
import pickle

warnings.filterwarnings("ignore")

# Configuration Settings
CONFIG = {
    "model_path": "yolov8n.pt",  # Default model path
    "confidence_threshold": 0.3,  # Confidence threshold for detections
    "sports": {
        "football": {
            "classes": [0],  # Person class in COCO dataset
            "actions": [
                "run",
                "pass",
                "shoot",
                "goal",
                "foul",
                "tackle",
                "dribble",
                "header",
                "corner",
                "throw_in",
            ],
            "field_dimensions": (105, 68),  # in meters (standard football field)
            "zones": {
                "defense": [(0, 0), (35, 68)],
                "midfield": [(35, 0), (70, 68)],
                "attack": [(70, 0), (105, 68)],
            },
        },
        "basketball": {
            "classes": [0],  # Person class in COCO dataset
            "actions": [
                "dribble",
                "pass",
                "jump_shot",
                "dunk",
                "foul",
                "block",
                "rebound",
                "steal",
                "layup",
                "screen",
            ],
            "field_dimensions": (28, 15),  # in meters (standard basketball court)
            "zones": {"defense": [(0, 0), (14, 15)], "attack": [(14, 0), (28, 15)]},
        },
        "cricket": {
            "classes": [0],  # Person class in COCO dataset
            "actions": [
                "bat",
                "bowl",
                "field",
                "catch",
                "run",
                "wicket",
                "throw",
                "appeal",
                "six",
                "four",
            ],
            "field_dimensions": (150, 150),  # in meters (circular field, approximation)
            "zones": {
                "pitch": [(70, 70), (80, 80)],  # Approximate pitch center
                "infield": [(50, 50), (100, 100)],
                "outfield": [(0, 0), (150, 150)],
            },
        },
        "volleyball": {
            "classes": [0],  # Person class in COCO dataset
            "actions": [
                "serve",
                "spike",
                "block",
                "dig",
                "set",
                "receive",
                "jump",
                "dive",
            ],
            "field_dimensions": (18, 9),  # in meters (standard volleyball court)
            "zones": {
                "front": [(0, 0), (9, 3)],
                "middle": [(0, 3), (9, 6)],
                "back": [(0, 6), (9, 9)],
            },
        },
        "hockey": {
            "classes": [0],  # Person class in COCO dataset
            "actions": ["pass", "shoot", "save", "tackle", "dribble", "penalty"],
            "field_dimensions": (91.4, 55),  # in meters (standard hockey field)
            "zones": {
                "defense": [(0, 0), (23, 55)],
                "midfield": [(23, 0), (68, 55)],
                "attack": [(68, 0), (91.4, 55)],
            },
        },
    },
    "output_dir": "output_data",
    "heatmap_resolution": (100, 100),
    "match_phases": ["first_half", "second_half", "extra_time", "penalties"],
    "max_fps_processing": 30,  # Maximum FPS for processing
    "default_ui_theme": "light",  # light or dark
    "camera_settings": {
        "webcam": {"default_device": 0, "resolution": (1280, 720)},
        "ip_camera": {
            "default_url": "http://192.168.1.100:8080/video",
            "username": "",
            "password": "",
        },
    },
    "advanced_analysis": {
        "formation_detection": True,
        "possession_analysis": True,
        "team_shape_analysis": True,
        "player_role_detection": True,
        "event_detection": True,
    },
    "visualization_settings": {
        "default_colormap": "viridis",
        "field_overlay_opacity": 0.7,
        "trajectory_line_width": 2,
        "marker_size": 8,
        "font_size": 10,
    },
}

# Create output directories
os.makedirs(CONFIG["output_dir"], exist_ok=True)
os.makedirs(os.path.join(CONFIG["output_dir"], "videos"), exist_ok=True)
os.makedirs(os.path.join(CONFIG["output_dir"], "data"), exist_ok=True)
os.makedirs(os.path.join(CONFIG["output_dir"], "visualizations"), exist_ok=True)
os.makedirs(os.path.join(CONFIG["output_dir"], "reports"), exist_ok=True)


class VideoSource:
    """Class to handle different video sources (file, webcam, IP camera)"""

    def __init__(
        self,
        source_type="file",
        source_path=None,
        ip_address=None,
        username=None,
        password=None,
        resolution=(1280, 720),
    ):
        """
        Initialize video source

        Args:
            source_type (str): Type of video source ("file", "webcam", "ip_camera")
            source_path (str): Path to video file or webcam index
            ip_address (str): IP camera address
            username (str): IP camera username
            password (str): IP camera password
            resolution (tuple): Desired resolution (width, height)
        """
        self.source_type = source_type
        self.source_path = source_path
        self.ip_address = ip_address
        self.username = username
        self.password = password
        self.resolution = resolution
        self.cap = None
        self.is_connected = False
        self.frame_width = None
        self.frame_height = None
        self.fps = None

    def connect(self):
        """Connect to the video source"""
        try:
            if self.source_type == "file" and self.source_path:
                self.cap = cv2.VideoCapture(self.source_path)
            elif self.source_type == "webcam":
                device_index = 0 if self.source_path is None else int(self.source_path)
                self.cap = cv2.VideoCapture(device_index)
                self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.resolution[0])
                self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.resolution[1])
            elif self.source_type == "ip_camera" and self.ip_address:
                # Construct URL based on authentication
                url = self.ip_address
                if self.username and self.password:
                    # Insert username/password into URL if needed
                    match = re.match(r"(https?://)(.+)", url)
                    if match:
                        protocol, rest = match.groups()
                        url = f"{protocol}{self.username}:{self.password}@{rest}"

                self.cap = cv2.VideoCapture(url)
            else:
                return False

            if not self.cap.isOpened():
                return False

            # Get video properties
            self.frame_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            self.frame_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            self.fps = self.cap.get(cv2.CAP_PROP_FPS)
            if self.fps <= 0 or self.fps is None:
                self.fps = 30  # Default FPS if not available

            self.is_connected = True
            return True

        except Exception as e:
            print(f"Error connecting to video source: {e}")
            return False

    def read_frame(self):
        """Read a frame from the video source"""
        if not self.is_connected or self.cap is None:
            return False, None

        return self.cap.read()

    def release(self):
        """Release the video source"""
        if self.cap is not None:
            self.cap.release()
            self.is_connected = False

    def get_video_info(self):
        """Get information about the video source"""
        if not self.is_connected:
            return None

        return {
            "source_type": self.source_type,
            "width": self.frame_width,
            "height": self.frame_height,
            "fps": self.fps,
            "total_frames": (
                int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
                if self.source_type == "file"
                else None
            ),
            "duration": (
                int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT) / self.fps)
                if self.source_type == "file"
                else None
            ),
        }


class TeamDetector:
    """Class to detect and classify teams in a sports video"""

    def __init__(self, clustering_algorithm="dbscan"):
        """
        Initialize team detector

        Args:
            clustering_algorithm (str): Algorithm to use for team classification
        """
        self.clustering_algorithm = clustering_algorithm
        self.team_colors = None
        self.is_trained = False

    def train_on_frame(self, frame, player_boxes):
        """
        Train team detector on a single frame

        Args:
            frame (np.ndarray): Video frame
            player_boxes (list): List of player bounding boxes in format [x1, y1, x2, y2]

        Returns:
            bool: Success status
        """
        if len(player_boxes) < 4:  # Need at least 4 players to detect teams
            return False

        # Extract player patches and their colors
        player_colors = []
        for box in player_boxes:
            x1, y1, x2, y2 = [int(val) for val in box]
            player_patch = frame[y1:y2, x1:x2]
            if player_patch.size == 0:
                continue

            # Calculate average color in HSV space (better for color clustering)
            player_patch_hsv = cv2.cvtColor(player_patch, cv2.COLOR_BGR2HSV)
            avg_h = np.mean(player_patch_hsv[:, :, 0])
            avg_s = np.mean(player_patch_hsv[:, :, 1])
            avg_v = np.mean(player_patch_hsv[:, :, 2])

            player_colors.append([avg_h, avg_s, avg_v])

        if len(player_colors) < 4:
            return False

        # Use clustering to identify teams
        if self.clustering_algorithm == "dbscan":
            clustering = DBSCAN(eps=25, min_samples=2).fit(player_colors)
            labels = clustering.labels_
        else:
            # K-means with k=2 for two teams
            from sklearn.cluster import KMeans

            clustering = KMeans(n_clusters=2).fit(player_colors)
            labels = clustering.labels_

        # Check if we have at least 2 clusters (2 teams)
        unique_labels = np.unique(labels)
        if len(unique_labels) < 2:
            return False

        # Calculate team colors (average color for each team)
        self.team_colors = []
        for label in unique_labels:
            if label == -1:  # Noise in DBSCAN
                continue

            team_color = np.mean(
                [
                    player_colors[i]
                    for i in range(len(player_colors))
                    if labels[i] == label
                ],
                axis=0,
            )
            self.team_colors.append(team_color)

        self.is_trained = True
        return True

    def classify_player(self, frame, player_box):
        """
        Classify a player's team

        Args:
            frame (np.ndarray): Video frame
            player_box (list): Player bounding box [x1, y1, x2, y2]

        Returns:
            int: Team index (0 or 1) or -1 if undetermined
        """
        if not self.is_trained or self.team_colors is None:
            return -1

        x1, y1, x2, y2 = [int(val) for val in player_box]
        player_patch = frame[y1:y2, x1:x2]
        if player_patch.size == 0:
            return -1

        # Calculate player color
        player_patch_hsv = cv2.cvtColor(player_patch, cv2.COLOR_BGR2HSV)
        avg_h = np.mean(player_patch_hsv[:, :, 0])
        avg_s = np.mean(player_patch_hsv[:, :, 1])
        avg_v = np.mean(player_patch_hsv[:, :, 2])
        player_color = np.array([avg_h, avg_s, avg_v])

        # Find closest team by color distance
        distances = [
            np.linalg.norm(player_color - team_color) for team_color in self.team_colors
        ]
        closest_team = np.argmin(distances)

        # Return team if distance is below threshold, otherwise undetermined
        if distances[closest_team] < 50:  # Color distance threshold
            return closest_team
        else:
            return -1

    def save_model(self, filename):
        """Save the team detector model"""
        if not self.is_trained:
            return False

        with open(filename, "wb") as f:
            pickle.dump(self.team_colors, f)
        return True

    def load_model(self, filename):
        """Load the team detector model"""
        try:
            with open(filename, "rb") as f:
                self.team_colors = pickle.load(f)
            self.is_trained = True
            return True
        except Exception as e:
            print(f"Error loading team detector model: {e}")
            return False


class PlayerTracker:
    def __init__(self, sport="football", model_path=None):
        """Initialize the player tracker with YOLOv8 model"""
        self.sport = sport
        self.sport_config = CONFIG["sports"][sport]

        # Load YOLOv8 model
        model_path = model_path or CONFIG["model_path"]
        self.model = YOLO(model_path)

        # Initialize player tracking data
        self.players = {}  # Dictionary to store player tracks
        self.player_speeds = {}
        self.player_distances = {}
        self.player_actions = {}
        self.player_positions = defaultdict(list)
        self.player_teams = {}  # To store team assignments
        self.frame_rate = None
        self.field_scale = None
        self.homography_matrix = None
        self.team_detector = TeamDetector()
        self.events = []  # Store detected events with timestamps
        self.possession_data = defaultdict(list)  # Store possession data by frames
        self.zone_data = defaultdict(
            lambda: defaultdict(int)
        )  # Store time spent in zones
        self.current_frame = 0
        self.action_detector = ActionDetector(sport)
        self.formation_detector = FormationDetector()

    def set_field_scale(self, video_dimensions, field_dimensions=None):
        """Set the scale to convert pixel distances to real-world distances"""
        field_dimensions = field_dimensions or self.sport_config["field_dimensions"]
        video_width, video_height = video_dimensions
        field_width, field_height = field_dimensions

        # Calculate scale factors (meters per pixel)
        self.field_scale = (field_width / video_width, field_height / video_height)

    def set_homography_matrix(self, source_points, destination_points):
        """Set the homography matrix to transform between image and field coordinates"""
        if len(source_points) >= 4 and len(destination_points) >= 4:
            self.homography_matrix = cv2.findHomography(
                np.array(source_points), np.array(destination_points)
            )[0]
        else:
            print("Warning: At least 4 points required for homography calculation")

    def pixel_to_field_coords(self, point):
        """Convert pixel coordinates to field coordinates"""
        if self.homography_matrix is not None:
            # Use homography for more accurate transformation
            px, py = point
            transformed_point = cv2.perspectiveTransform(
                np.array([[[px, py]]], dtype=np.float32), self.homography_matrix
            )
            return (transformed_point[0][0][0], transformed_point[0][0][1])
        elif self.field_scale is not None:
            # Use simple scaling
            x_scale, y_scale = self.field_scale
            return (point[0] * x_scale, point[1] * y_scale)
        else:
            # Return original coordinates if no transformation is set
            return point

    def calculate_player_zone(self, position):
        """
        Calculate which zone a player is in based on their position

        Args:
            position (tuple): Player position (x, y) in field coordinates

        Returns:
            str: Zone name
        """
        x, y = position
        zones = self.sport_config["zones"]

        for zone_name, ((x1, y1), (x2, y2)) in zones.items():
            if x1 <= x <= x2 and y1 <= y <= y2:
                return zone_name

        return "unknown"

    def process_frame(self, frame, detect_teams=True):
        """
        Process a single frame to track players

        Args:
            frame (np.ndarray): Video frame
            detect_teams (bool): Whether to detect teams

        Returns:
            tuple: (annotated_frame, tracking_results)
        """
        if frame is None:
            return None, None

        self.current_frame += 1

        # Run YOLOv8 detection and tracking
        results = self.model.track(
            frame,
            persist=True,
            classes=self.sport_config["classes"],
            conf=CONFIG["confidence_threshold"],
        )

        # Initial frame for team detection
        if (
            detect_teams
            and not self.team_detector.is_trained
            and results[0].boxes is not None
        ):
            boxes = results[0].boxes.xyxy.cpu().numpy()
            if len(boxes) >= 6:  # Need enough players to detect teams
                self.team_detector.train_on_frame(frame, boxes)

        # Extract tracking data
        if (
            results[0].boxes is not None
            and hasattr(results[0].boxes, "id")
            and results[0].boxes.id is not None
        ):
            boxes = results[0].boxes.xyxy.cpu().numpy()
            track_ids = results[0].boxes.id.int().cpu().numpy()

            # Team detection for each player
            if detect_teams and self.team_detector.is_trained:
                for i, box in enumerate(boxes):
                    track_id = track_ids[i]
                    team = self.team_detector.classify_player(frame, box)

                    if track_id not in self.player_teams:
                        self.player_teams[track_id] = team
                    else:
                        # Update team assignment with moving average
                        if team != -1:  # Only update if team is detected
                            current_team = self.player_teams[track_id]
                            if current_team == -1:
                                self.player_teams[track_id] = team
                            # Else keep the team assignment

            # Process each detected player
            for i, box in enumerate(boxes):
                track_id = track_ids[i]
                x1, y1, x2, y2 = box

                # Calculate center point
                center_x = (x1 + x2) / 2
                center_y = (y1 + y2) / 2

                # Convert to field coordinates
                field_position = self.pixel_to_field_coords((center_x, center_y))

                # Store position history
                if track_id not in self.players:
                    self.players[track_id] = []
                    self.player_speeds[track_id] = []
                    self.player_distances[track_id] = 0
                    self.player_actions[track_id] = []

                # Calculate speed and distance if we have previous positions
                if len(self.players[track_id]) > 0:
                    prev_position = self.players[track_id][-1]
                    dist = distance.euclidean(field_position, prev_position)

                    # Calculate speed (distance / time)
                    time_diff = (
                        1 / self.frame_rate if self.frame_rate else 1 / 30
                    )  # time between frames
                    speed = dist / time_diff  # in meters per second

                    self.player_speeds[track_id].append(speed)
                    self.player_distances[track_id] += dist

                # Store current position
                self.players[track_id].append(field_position)
                self.player_positions[self.current_frame].append(
                    (track_id, field_position)
                )

                # Track player zones
                zone = self.calculate_player_zone(field_position)
                self.zone_data[track_id][zone] += 1

            # Detect actions
            actions = self.action_detector.detect_actions(
                frame, boxes, track_ids, self.players, self.player_speeds
            )
            for track_id, action in actions:
                self.player_actions[track_id].append((self.current_frame, action))

                # Add to events list if significant
                if action in ["goal", "foul", "tackle", "dunk", "wicket"]:
                    self.events.append(
                        {
                            "frame": self.current_frame,
                            "time": (
                                self.current_frame / self.frame_rate
                                if self.frame_rate
                                else 0
                            ),
                            "player": track_id,
                            "action": action,
                            "position": (
                                self.players[track_id][-1]
                                if self.players[track_id]
                                else None
                            ),
                            "team": self.player_teams.get(track_id, -1),
                        }
                    )

            # Update formation if enough players
            if len(boxes) >= 6:
                team_positions = defaultdict(list)
                for i, track_id in enumerate(track_ids):
                    team = self.player_teams.get(track_id, -1)
                    if team != -1:
                        pos = (
                            self.players[track_id][-1]
                            if self.players[track_id]
                            else None
                        )
                        if pos:
                            team_positions[team].append((track_id, pos))

                # Detect formations for each team
                for team, positions in team_positions.items():
                    if (
                        len(positions) >= 5
                    ):  # Need at least 5 players to detect formation
                        formation = self.formation_detector.detect_formation(
                            [pos for _, pos in positions],
                            self.sport,
                            team_positions,  # Pass all team positions for context
                        )
                        if formation:
                            # Add formation change to events if significant
                            self.events.append(
                                {
                                    "frame": self.current_frame,
                                    "time": (
                                        self.current_frame / self.frame_rate
                                        if self.frame_rate
                                        else 0
                                    ),
                                    "team": team,
                                    "action": "formation_change",
                                    "formation": formation,
                                }
                            )

            # Update possession data
            self.update_possession(boxes, track_ids)

        # Create annotated frame
        annotated_frame = results[0].plot()

        # Draw additional info like teams, speeds, zones
        if hasattr(results[0].boxes, "id") and results[0].boxes.id is not None:
            boxes = results[0].boxes.xyxy.cpu().numpy()
            track_ids = results[0].boxes.id.int().cpu().numpy()

            for i, box in enumerate(boxes):
                track_id = track_ids[i]
                x1, y1, x2, y2 = [int(val) for val in box]

                # Draw team indicator
                team = self.player_teams.get(track_id, -1)
                team_color = (
                    (0, 0, 255)
                    if team == 0
                    else (255, 0, 0) if team == 1 else (200, 200, 200)
                )
                cv2.rectangle(
                    annotated_frame, (x1, y1 - 25), (x1 + 20, y1 - 5), team_color, -1
                )

                # Show player ID and speed
                if (
                    track_id in self.player_speeds
                    and len(self.player_speeds[track_id]) > 0
                ):
                    speed = self.player_speeds[track_id][-1]
                    cv2.putText(
                        annotated_frame,
                        f"ID:{track_id} {speed:.1f}m/s",
                        (x1, y1 - 8),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        (255, 255, 255),
                        2,
                    )

                # Show recent action if available
                recent_actions = [
                    action
                    for frame, action in self.player_actions.get(track_id, [])
                    if self.current_frame - frame < 30
                ]
                if recent_actions:
                    cv2.putText(
                        annotated_frame,
                        recent_actions[-1],
                        (x1, y2 + 15),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        (0, 255, 0),
                        2,
                    )

        return annotated_frame, results

    def update_possession(self, boxes, track_ids):
        """
        Update possession statistics based on current frame

        Args:
            boxes (np.ndarray): Bounding boxes
            track_ids (np.ndarray): Track IDs
        """
        if len(boxes) == 0:
            return

        team_count = {0: 0, 1: 0}
        zone_control = defaultdict(lambda: {0: 0, 1: 0})

        # Count players in each zone by team
        for i, track_id in enumerate(track_ids):
            if track_id in self.players and len(self.players[track_id]) > 0:
                team = self.player_teams.get(track_id, -1)
                if team != -1:
                    team_count[team] += 1

                    # Update zone control
                    position = self.players[track_id][-1]
                    zone = self.calculate_player_zone(position)
                    zone_control[zone][team] += 1

        # Determine possession based on player positions and ball control (simplified)
        if team_count[0] > 0 and team_count[1] > 0:
            # For simplicity, assume team with more players in attack zone has possession
            attack_zone = "attack"
            if attack_zone in zone_control:
                if zone_control[attack_zone][0] > zone_control[attack_zone][1]:
                    possession_team = 0
                elif zone_control[attack_zone][1] > zone_control[attack_zone][0]:
                    possession_team = 1
                else:
                    # If equal, check midfield
                    midfield_zone = "midfield"
                    if midfield_zone in zone_control:
                        if (
                            zone_control[midfield_zone][0]
                            > zone_control[midfield_zone][1]
                        ):
                            possession_team = 0
                        elif (
                            zone_control[midfield_zone][1]
                            > zone_control[midfield_zone][0]
                        ):
                            possession_team = 1
                        else:
                            possession_team = -1  # Undetermined
                    else:
                        possession_team = -1
            else:
                possession_team = -1

            # Store possession data
            self.possession_data[self.current_frame] = {
                "team": possession_team,
                "team_count": dict(team_count),
                "zone_control": {k: dict(v) for k, v in zone_control.items()},
            }

    def process_video(
        self,
        video_path,
        output_path=None,
        save_frames=False,
        detect_teams=True,
        real_time=False,
        progress_callback=None,
    ):
        """
        Process a video file to track players and analyze movements

        Args:
            video_path (str): Path to video file
            output_path (str): Path to save output video
            save_frames (bool): Whether to save individual frames
            detect_teams (bool): Whether to detect teams
            real_time (bool): Whether to process in real-time
            progress_callback (function): Callback function for progress updates

        Returns:
            dict: Analysis results
        """
        video_source = VideoSource(source_type="file", source_path=video_path)
        if not video_source.connect():
            raise ValueError(f"Could not open video file: {video_path}")

        # Get video properties
        video_info = video_source.get_video_info()
        self.frame_width = video_info["width"]
        self.frame_height = video_info["height"]
        self.frame_rate = video_info["fps"]
        total_frames = video_info["total_frames"]

        # Set field scale if not already set
        if self.field_scale is None:
            self.set_field_scale((self.frame_width, self.frame_height))

        # Setup output video if required
        if output_path:
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            out = cv2.VideoWriter(
                output_path,
                fourcc,
                self.frame_rate,
                (self.frame_width, self.frame_height),
            )

        frame_count = 0
        last_percent = -1
        processing_start_time = time.time()

        # For real-time processing rate control
        frame_time = 1.0 / CONFIG["max_fps_processing"] if real_time else 0

        # Process frames
        while True:
            ret, frame = video_source.read_frame()
            if not ret:
                break

            # Progress indication
            frame_count += 1
            if total_frames:
                percent_complete = int((frame_count / total_frames) * 100)
                if percent_complete != last_percent and percent_complete % 5 == 0:
                    if progress_callback:
                        progress_callback(percent_complete)
                    else:
                        print(f"Processing video: {percent_complete}% complete")
                    last_percent = percent_complete

            # Process frame
            start_time = time.time()
            annotated_frame, results = self.process_frame(frame, detect_teams)

            if annotated_frame is None:
                continue

            # Save outputs
            if output_path:
                out.write(annotated_frame)

            if save_frames:
                frames_dir = os.path.join(CONFIG["output_dir"], "frames")
                os.makedirs(frames_dir, exist_ok=True)
                cv2.imwrite(
                    os.path.join(frames_dir, f"frame_{frame_count:04d}.jpg"),
                    annotated_frame,
                )

            # Real-time processing rate control
            if real_time:
                processing_time = time.time() - start_time
                if processing_time < frame_time:
                    time.sleep(frame_time - processing_time)

        # Clean up
        video_source.release()
        if output_path:
            out.release()

        processing_duration = time.time() - processing_start_time

        # Return analysis results
        analysis_results = {
            "player_tracks": self.players,
            "player_speeds": self.player_speeds,
            "player_distances": self.player_distances,
            "player_actions": self.player_actions,
            "frame_positions": self.player_positions,
            "player_teams": self.player_teams,
            "events": self.events,
            "possession_data": self.possession_data,
            "zone_data": self.zone_data,
            "total_frames": frame_count,
            "frame_rate": self.frame_rate,
            "processing_duration": processing_duration,
        }

        # Save analysis results
        results_path = os.path.join(
            CONFIG["output_dir"], "data", f"analysis_{int(time.time())}.pkl"
        )
        with open(results_path, "wb") as f:
            pickle.dump(analysis_results, f)

        return analysis_results

    def process_stream(
        self,
        video_source,
        output_frame_queue,
        stop_event,
        detect_teams=True,
        save_output=False,
        output_path=None,
    ):
        """
        Process a video stream (webcam or IP camera)

        Args:
            video_source (VideoSource): Video source object
            output_frame_queue (Queue): Queue to store output frames
            stop_event (Event): Event to signal stopping
            detect_teams (bool): Whether to detect teams
            save_output (bool): Whether to save output video
            output_path (str): Path to save output video
        """
        if not video_source.is_connected and not video_source.connect():
            print("Could not connect to video source")
            return

        # Get video properties
        video_info = video_source.get_video_info()
        self.frame_width = video_info["width"]
        self.frame_height = video_info["height"]
        self.frame_rate = video_info["fps"]

        # Set field scale if not already set
        if self.field_scale is None:
            self.set_field_scale((self.frame_width, self.frame_height))

        # Setup output video if required
        if save_output and output_path:
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            out = cv2.VideoWriter(
                output_path,
                fourcc,
                self.frame_rate,
                (self.frame_width, self.frame_height),
            )
        else:
            out = None

        frame_time = 1.0 / CONFIG["max_fps_processing"]

        # Reset tracking data for stream
        self.players = {}
        self.player_speeds = {}
        self.player_distances = {}
        self.player_actions = {}
        self.player_positions = defaultdict(list)
        self.player_teams = {}
        self.events = []
        self.possession_data = defaultdict(list)
        self.zone_data = defaultdict(lambda: defaultdict(int))
        self.current_frame = 0

        # Process frames
        while not stop_event.is_set():
            start_time = time.time()
            ret, frame = video_source.read_frame()
            if not ret:
                # Try to reconnect for IP cameras that might have temporary issues
                if video_source.source_type == "ip_camera":
                    print("Lost connection to IP camera. Attempting to reconnect...")
                    if video_source.connect():
                        continue
                break

            # Process frame
            annotated_frame, results = self.process_frame(frame, detect_teams)

            if annotated_frame is None:
                continue

            # Add analysis overlay
            annotated_frame = self.add_analysis_overlay(annotated_frame)

            # Put frame in output queue for display
            if output_frame_queue and not output_frame_queue.full():
                output_frame_queue.put(annotated_frame)

            # Save output if required
            if out:
                out.write(annotated_frame)

            # Control processing rate
            processing_time = time.time() - start_time
            if processing_time < frame_time:
                time.sleep(frame_time - processing_time)

        # Clean up
        video_source.release()
        if out:
            out.release()

        # Save analysis results
        if self.current_frame > 0:
            # Create a deep copy of data to avoid saving any lambda or unpicklable objects
            cleaned_events = []
            for event in self.events:
                # Create a new dictionary without any lambda functions
                cleaned_event = {}
                for k, v in event.items():
                    if not callable(v):  # Skip callable objects like lambdas
                        cleaned_event[k] = v
                cleaned_events.append(cleaned_event)

            # Create cleaned possession data
            cleaned_possession = {}
            for frame, data in self.possession_data.items():
                cleaned_possession[frame] = {}
                for k, v in data.items():
                    if isinstance(v, dict):
                        cleaned_possession[frame][k] = dict(v)
                    elif not callable(v):
                        cleaned_possession[frame][k] = v

            analysis_results = {
                "player_tracks": dict(self.players),
                "player_speeds": dict(self.player_speeds),
                "player_distances": dict(self.player_distances),
                "player_actions": dict(self.player_actions),
                "player_teams": dict(self.player_teams),
                "events": cleaned_events,
                "possession_data": cleaned_possession,
                "zone_data": dict(self.zone_data),
                "frame_positions": dict(self.player_positions),
                "total_frames": self.current_frame,
                "frame_rate": self.frame_rate,
            }

            results_path = os.path.join(
                CONFIG["output_dir"], "data", f"stream_analysis_{int(time.time())}.pkl"
            )
            try:
                with open(results_path, "wb") as f:
                    pickle.dump(analysis_results, f)
            except Exception as e:
                print(f"Error saving analysis data: {e}")

    def add_analysis_overlay(self, frame):
        """
        Add real-time analysis overlay to frame

        Args:
            frame (np.ndarray): Input frame

        Returns:
            np.ndarray: Frame with overlay
        """
        # Create transparent overlay
        overlay = frame.copy()

        # Add possession indicator
        if self.current_frame in self.possession_data:
            possession = self.possession_data[self.current_frame]
            team = possession["team"]

            if team != -1:
                team_name = f"Team {team+1}"
                team_color = (0, 0, 255) if team == 0 else (255, 0, 0)

                # Draw possession bar at top
                cv2.rectangle(overlay, (0, 0), (frame.shape[1], 40), (0, 0, 0), -1)
                cv2.putText(
                    overlay,
                    f"Possession: {team_name}",
                    (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    team_color,
                    2,
                )

                # Show possession percentage
                team_0_frames = sum(
                    1
                    for f in self.possession_data
                    if self.possession_data[f]["team"] == 0
                )
                team_1_frames = sum(
                    1
                    for f in self.possession_data
                    if self.possession_data[f]["team"] == 1
                )
                total_possession_frames = team_0_frames + team_1_frames

                if total_possession_frames > 0:
                    team_0_percent = int(team_0_frames / total_possession_frames * 100)
                    team_1_percent = int(team_1_frames / total_possession_frames * 100)

                    cv2.putText(
                        overlay,
                        f"Possession %: Team 1: {team_0_percent}% | Team 2: {team_1_percent}%",
                        (frame.shape[1] // 3, 30),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7,
                        (255, 255, 255),
                        2,
                    )

        # Add recent events
        recent_events = [
            e for e in self.events if self.current_frame - e["frame"] < 150
        ]  # Last 5 seconds
        if recent_events:
            y_pos = 70
            cv2.rectangle(
                overlay, (0, 40), (300, 40 + 30 * len(recent_events)), (0, 0, 0), -1
            )

            for event in recent_events:
                event_text = f"{event['action'].upper()}"
                if "player" in event:
                    event_text += f" - Player {event['player']}"
                if "team" in event and event["team"] != -1:
                    event_text += f" - Team {event['team']+1}"

                cv2.putText(
                    overlay,
                    event_text,
                    (10, y_pos),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 255, 0),
                    2,
                )
                y_pos += 30

        # Add frame count and timestamp
        time_str = (
            str(timedelta(seconds=int(self.current_frame / self.frame_rate)))
            if self.frame_rate
            else ""
        )
        cv2.putText(
            overlay,
            f"Frame: {self.current_frame} | Time: {time_str}",
            (frame.shape[1] - 350, frame.shape[0] - 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            2,
        )

        # Blend overlay with original frame
        alpha = 0.7
        cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)

        return frame

    def save_tracking_data(self, filename):
        """Save tracking data to a CSV file"""
        data = []

        for player_id, positions in self.players.items():
            team = self.player_teams.get(player_id, -1)

            for i, (x, y) in enumerate(positions):
                speed = (
                    self.player_speeds[player_id][i]
                    if i < len(self.player_speeds[player_id])
                    else None
                )

                data.append(
                    {
                        "player_id": player_id,
                        "team": team,
                        "position_index": i,
                        "x": x,
                        "y": y,
                        "speed": speed,
                    }
                )

        # Convert to DataFrame and save
        df = pd.DataFrame(data)
        df.to_csv(filename, index=False)

        return df

    def get_player_stats(self, player_id):
        """Get comprehensive stats for a specific player"""
        if player_id not in self.players:
            return None

        positions = self.players[player_id]
        speeds = (
            self.player_speeds[player_id] if player_id in self.player_speeds else []
        )
        distance = (
            self.player_distances[player_id]
            if player_id in self.player_distances
            else 0
        )
        actions = (
            self.player_actions[player_id] if player_id in self.player_actions else []
        )
        team = self.player_teams.get(player_id, -1)

        # Calculate statistics
        avg_speed = sum(speeds) / len(speeds) if speeds else 0
        max_speed = max(speeds) if speeds else 0

        # Calculate time spent in different zones
        total_frames = sum(self.zone_data[player_id].values())
        zone_percentages = {}
        for zone, count in self.zone_data[player_id].items():
            if total_frames > 0:
                zone_percentages[zone] = (count / total_frames) * 100
            else:
                zone_percentages[zone] = 0

        # Count actions
        action_counts = defaultdict(int)
        for _, action in actions:
            action_counts[action] += 1

        # Calculate heatmap data for this player
        heatmap_data = self.generate_heatmap_data(player_id)

        return {
            "player_id": player_id,
            "team": team,
            "total_distance": distance,
            "avg_speed": avg_speed,
            "max_speed": max_speed,
            "action_counts": dict(action_counts),
            "position_count": len(positions),
            "zone_percentages": zone_percentages,
            "heatmap_data": heatmap_data,
        }

    def generate_heatmap_data(self, player_id=None, team=None):
        """Generate heatmap data for a specific player, team, or all players"""
        heatmap_data = np.zeros(CONFIG["heatmap_resolution"])

        # Get field dimensions
        field_width, field_height = self.sport_config["field_dimensions"]
        x_bins = CONFIG["heatmap_resolution"][0]
        y_bins = CONFIG["heatmap_resolution"][1]

        # Filter positions based on player_id or team
        if player_id is not None and player_id in self.players:
            positions = self.players[player_id]
        elif team is not None:
            # Combine positions for all players in the team
            positions = []
            for pid, player_positions in self.players.items():
                if self.player_teams.get(pid, -1) == team:
                    positions.extend(player_positions)
        else:
            # Combine all player positions
            positions = []
            for player_positions in self.players.values():
                positions.extend(player_positions)

        # Create heatmap data
        for x, y in positions:
            # Skip positions outside the field
            if x < 0 or x > field_width or y < 0 or y > field_height:
                continue

            # Convert position to bin indices
            x_idx = min(int((x / field_width) * x_bins), x_bins - 1)
            y_idx = min(int((y / field_height) * y_bins), y_bins - 1)

            # Increment bin count
            heatmap_data[y_idx, x_idx] += 1

        return heatmap_data

    def get_team_stats(self, team):
        """Get comprehensive stats for a team"""
        if team not in [0, 1]:
            return None

        # Get all players in the team
        team_players = [pid for pid, t in self.player_teams.items() if t == team]

        if not team_players:
            return None

        # Aggregate data
        total_distance = sum(self.player_distances.get(pid, 0) for pid in team_players)

        # Speeds
        all_speeds = []
        for pid in team_players:
            if pid in self.player_speeds:
                all_speeds.extend(self.player_speeds[pid])

        avg_speed = sum(all_speeds) / len(all_speeds) if all_speeds else 0
        max_speed = max(all_speeds) if all_speeds else 0

        # Actions
        action_counts = defaultdict(int)
        for pid in team_players:
            for _, action in self.player_actions.get(pid, []):
                action_counts[action] += 1

        # Calculate heatmap data for the team
        heatmap_data = self.generate_heatmap_data(team=team)

        # Calculate possession percentage
        team_possession_frames = sum(
            1 for f in self.possession_data if self.possession_data[f]["team"] == team
        )
        total_possession_frames = sum(1 for f in self.possession_data)
        possession_percentage = (
            (team_possession_frames / total_possession_frames * 100)
            if total_possession_frames
            else 0
        )

        # Calculate zone control
        zone_control = defaultdict(float)
        for frame, data in self.possession_data.items():
            for zone, control in data.get("zone_control", {}).items():
                if team in control:
                    zone_control[zone] += control[team]

        # Average zone control
        for zone in zone_control:
            zone_control[zone] /= (
                total_possession_frames if total_possession_frames else 1
            )

        return {
            "team": team,
            "player_count": len(team_players),
            "total_distance": total_distance,
            "avg_speed": avg_speed,
            "max_speed": max_speed,
            "action_counts": dict(action_counts),
            "possession_percentage": possession_percentage,
            "zone_control": dict(zone_control),
            "heatmap_data": heatmap_data,
        }

    def analyze_interactions(self, distance_threshold=5):
        """Analyze player interactions based on proximity"""
        interactions = defaultdict(int)
        team_interactions = {
            (0, 0): 0,  # Team 0 internal interactions
            (1, 1): 0,  # Team 1 internal interactions
            (0, 1): 0,  # Cross-team interactions
        }

        # Process each frame's positions
        for frame, positions in self.player_positions.items():
            # Skip frames with fewer than 2 players
            if len(positions) < 2:
                continue

            # Check distance between every pair of players
            for i, (player1, pos1) in enumerate(positions):
                team1 = self.player_teams.get(player1, -1)

                for j, (player2, pos2) in enumerate(positions[i + 1 :], i + 1):
                    team2 = self.player_teams.get(player2, -1)
                    dist = distance.euclidean(pos1, pos2)

                    # Count interaction if players are within threshold
                    if dist <= distance_threshold:
                        # Sort player IDs to ensure consistent counting
                        pair = tuple(sorted([player1, player2]))
                        interactions[pair] += 1

                        # Count team interactions if both players have team assignments
                        if team1 != -1 and team2 != -1:
                            team_pair = tuple(sorted([team1, team2]))
                            team_interactions[team_pair] += 1

        return {
            "player_interactions": dict(interactions),
            "team_interactions": dict(team_interactions),
        }


class ActionDetector:
    def __init__(self, sport="football"):
        """Initialize action detector for a specific sport"""
        self.sport = sport
        self.sport_config = CONFIG["sports"][sport]
        self.history = defaultdict(list)  # Store recent positions and speeds
        self.history_size = 10  # Number of frames to keep in history

    def detect_actions(self, frame, boxes, track_ids, player_tracks, player_speeds):
        """
        Detect player actions based on movement patterns and posture

        Args:
            frame (np.ndarray): Video frame
            boxes (np.ndarray): Bounding boxes
            track_ids (np.ndarray): Track IDs
            player_tracks (dict): Dictionary of player tracks
            player_speeds (dict): Dictionary of player speeds

        Returns:
            list: List of (player_id, action) tuples
        """
        actions = []

        for i, box in enumerate(boxes):
            track_id = track_ids[i]
            x1, y1, x2, y2 = box

            # Skip if not enough history
            if track_id not in player_tracks or len(player_tracks[track_id]) < 3:
                continue

            # Update history
            self.update_history(
                track_id, player_tracks[track_id], player_speeds.get(track_id, [])
            )

            # Get player patch for posture analysis
            player_patch = frame[int(y1) : int(y2), int(x1) : int(x2)]

            # Detect actions based on sport
            if self.sport == "football":
                detected_action = self.detect_football_actions(track_id, player_patch)
            elif self.sport == "basketball":
                detected_action = self.detect_basketball_actions(track_id, player_patch)
            elif self.sport == "volleyball":
                detected_action = self.detect_volleyball_actions(track_id, player_patch)
            elif self.sport == "cricket":
                detected_action = self.detect_cricket_actions(track_id, player_patch)
            elif self.sport == "hockey":
                detected_action = self.detect_hockey_actions(track_id, player_patch)
            else:
                detected_action = None

            if detected_action:
                actions.append((track_id, detected_action))

        return actions

    def update_history(self, track_id, positions, speeds):
        """Update position and speed history for a player"""
        # Add latest position
        if positions:
            latest_pos = positions[-1]
            if track_id not in self.history or len(self.history[track_id]) == 0:
                self.history[track_id] = [(latest_pos, None)]
            elif latest_pos != self.history[track_id][-1][0]:  # Avoid duplicates
                # Add latest speed if available
                latest_speed = speeds[-1] if speeds else None
                self.history[track_id].append((latest_pos, latest_speed))

        # Trim history to keep only the most recent entries
        if len(self.history[track_id]) > self.history_size:
            self.history[track_id] = self.history[track_id][-self.history_size :]

    def detect_football_actions(self, track_id, player_patch):
        """Detect football-specific actions"""
        if track_id not in self.history or len(self.history[track_id]) < 3:
            return None

        # Get position and speed history
        positions = [pos for pos, _ in self.history[track_id]]
        speeds = [speed for _, speed in self.history[track_id] if speed is not None]

        # Calculate movement features
        if len(speeds) >= 2:
            avg_speed = sum(speeds) / len(speeds)
            max_speed = max(speeds)
            acceleration = speeds[-1] - speeds[0]

            # Check for sprint
            if max_speed > 7 and acceleration > 1:
                return "run"

            # Check for direction change (potential tackle or dribble)
            if self._check_direction_change(positions):
                if avg_speed > 5:
                    return "tackle"
                elif 3 < avg_speed < 5:
                    return "dribble"

            # Check for sudden stop (potential shot)
            if (
                len(speeds) >= 3
                and speeds[-3] > speeds[-2] > speeds[-1]
                and speeds[-3] - speeds[-1] > 2
            ):
                return "shoot"

        # Check for posture-based actions using the player patch
        if player_patch.size > 0:
            # Simple analysis of player posture (height-to-width ratio)
            h, w = player_patch.shape[:2]
            ratio = h / w if w > 0 else 0

            # Jumping/heading detection (high height-to-width ratio)
            if ratio > 3:
                return "header"

        return None

    def detect_basketball_actions(self, track_id, player_patch):
        """Detect basketball-specific actions"""
        if track_id not in self.history or len(self.history[track_id]) < 3:
            return None

        # Get position history
        positions = [pos for pos, _ in self.history[track_id]]
        speeds = [speed for _, speed in self.history[track_id] if speed is not None]

        # Check for jump (vertical movement)
        if self._check_vertical_movement(positions):
            return "jump_shot"

        # Check for sudden direction changes (dribble)
        if self._check_direction_change(positions):
            return "dribble"

        # Check for posture-based actions
        if player_patch.size > 0:
            h, w = player_patch.shape[:2]
            ratio = h / w if w > 0 else 0

            # Very high pose (potential dunk or block)
            if ratio > 3.5 and len(speeds) > 0 and speeds[-1] < 2:
                # Near the basket (simplified check)
                return "dunk"

        return None

    def detect_volleyball_actions(self, track_id, player_patch):
        """Detect volleyball-specific actions"""
        if track_id not in self.history or len(self.history[track_id]) < 3:
            return None

        # Get position history
        positions = [pos for pos, _ in self.history[track_id]]

        # Check for jump (vertical movement)
        if self._check_vertical_movement(positions):
            # Check arm position for spike vs. block
            if player_patch.size > 0:
                h, w = player_patch.shape[:2]
                # Simple heuristic: analyze upper part of the player
                upper_part = player_patch[: h // 2, :]
                # Calculate the horizontal spread (arms extended for block vs. raised for spike)
                horizontal_profile = np.sum(
                    upper_part > 30, axis=0
                )  # Threshold to find player silhouette
                spread = np.count_nonzero(horizontal_profile) / w if w > 0 else 0

                if spread > 0.7:  # Arms extended wide
                    return "block"
                else:
                    return "spike"

        # Check for dive
        if self._check_dive_movement(positions):
            return "dive"

        return None

    def detect_cricket_actions(self, track_id, player_patch):
        """Detect cricket-specific actions"""
        if track_id not in self.history or len(self.history[track_id]) < 3:
            return None

        # Get position and speed history
        positions = [pos for pos, _ in self.history[track_id]]
        speeds = [speed for _, speed in self.history[track_id] if speed is not None]

        # Check for running between wickets
        if len(speeds) > 0 and speeds[-1] > 5:
            return "run"

        # Check for batting or bowling posture
        if player_patch.size > 0:
            h, w = player_patch.shape[:2]
            ratio = h / w if w > 0 else 0

            # Wide stance (potential batting)
            if 1.5 < ratio < 2.5:
                return "bat"

            # Tall pose with arm movement (potential bowling)
            if ratio > 2.5:
                return "bowl"

        return None

    def detect_hockey_actions(self, track_id, player_patch):
        """Detect hockey-specific actions"""
        if track_id not in self.history or len(self.history[track_id]) < 3:
            return None

        # Get position and speed history
        positions = [pos for pos, _ in self.history[track_id]]
        speeds = [speed for _, speed in self.history[track_id] if speed is not None]

        # Check for sprint
        if len(speeds) > 0 and speeds[-1] > 6:
            return "run"

        # Check for sudden direction changes (dribble)
        if self._check_direction_change(positions):
            return "dribble"

        # Check for shooting posture
        if player_patch.size > 0:
            # Simple analysis based on player silhouette
            h, w = player_patch.shape[:2]
            # Check for characteristic shooting pose (simplified)
            if h > 0 and w > 0:
                # Convert to grayscale and threshold to get silhouette
                gray = (
                    cv2.cvtColor(player_patch, cv2.COLOR_BGR2GRAY)
                    if len(player_patch.shape) == 3
                    else player_patch
                )
                _, thresh = cv2.threshold(gray, 30, 255, cv2.THRESH_BINARY)

                # Calculate contour properties
                contours, _ = cv2.findContours(
                    thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
                )
                if contours:
                    # Get largest contour
                    largest_contour = max(contours, key=cv2.contourArea)
                    # Check contour shape for shooting posture
                    rect = cv2.minAreaRect(largest_contour)
                    _, (width, height), angle = rect

                    # Shooting often involves a rotated posture
                    if abs(angle) > 15 and abs(angle) < 75:
                        return "shoot"

        return None

    def _check_direction_change(self, positions):
        """Check if there's a significant direction change in recent positions"""
        if len(positions) < 3:
            return False

        # Calculate direction vectors
        vec1 = (
            positions[-2][0] - positions[-3][0],
            positions[-2][1] - positions[-3][1],
        )
        vec2 = (
            positions[-1][0] - positions[-2][0],
            positions[-1][1] - positions[-2][1],
        )

        # Normalize vectors
        mag1 = np.sqrt(vec1[0] ** 2 + vec1[1] ** 2)
        mag2 = np.sqrt(vec2[0] ** 2 + vec2[1] ** 2)

        if mag1 > 0 and mag2 > 0:
            vec1_norm = (vec1[0] / mag1, vec1[1] / mag1)
            vec2_norm = (vec2[0] / mag2, vec2[1] / mag2)

            # Calculate dot product to find angle
            dot_product = vec1_norm[0] * vec2_norm[0] + vec1_norm[1] * vec2_norm[1]

            # A value less than 0.5 indicates a significant direction change (> 60 degrees)
            return dot_product < 0.5

        return False

    def _check_vertical_movement(self, positions):
        """Check if there's significant vertical movement (for jump detection)"""
        if len(positions) < 4:
            return False

        # Check for a pattern of going up and then down
        y_positions = [pos[1] for pos in positions]

        # Simplified jump detection: decrease then increase in y position
        y_diffs = [
            y_positions[i + 1] - y_positions[i] for i in range(len(y_positions) - 1)
        ]

        # Look for a pattern of negative (going up) then positive (going down)
        if len(y_diffs) >= 2 and y_diffs[-2] < -0.2 and y_diffs[-1] > 0.2:
            return True

        return False

    def _check_dive_movement(self, positions):
        """Check for dive movement pattern"""
        if len(positions) < 4:
            return False

        # Check for horizontal movement followed by stopping
        x_positions = [pos[0] for pos in positions]
        y_positions = [pos[1] for pos in positions]

        # Calculate movements
        x_diffs = [
            x_positions[i + 1] - x_positions[i] for i in range(len(x_positions) - 1)
        ]
        y_diffs = [
            y_positions[i + 1] - y_positions[i] for i in range(len(y_positions) - 1)
        ]

        # Calculate speeds (simplified as displacement per frame)
        speeds = [
            np.sqrt(x_diffs[i] ** 2 + y_diffs[i] ** 2) for i in range(len(x_diffs))
        ]

        # Dive pattern: fast movement, then sudden stop, with horizontal component
        if len(speeds) >= 3:
            if speeds[-3] > 0.5 and speeds[-2] > 0.5 and speeds[-1] < 0.2:
                # Check if movement was primarily horizontal
                horizontal_ratio = sum(abs(x) for x in x_diffs[-3:]) / (
                    sum(abs(y) for y in y_diffs[-3:]) + 0.001
                )
                if horizontal_ratio > 1.5:
                    return True

        return False


class FormationDetector:
    def __init__(self):
        """Initialize formation detector"""
        self.known_formations = {
            "football": {
                "4-4-2": {"defenders": 4, "midfielders": 4, "forwards": 2},
                "4-3-3": {"defenders": 4, "midfielders": 3, "forwards": 3},
                "3-5-2": {"defenders": 3, "midfielders": 5, "forwards": 2},
                "5-3-2": {"defenders": 5, "midfielders": 3, "forwards": 2},
                "4-2-3-1": {
                    "defenders": 4,
                    "defensive_mid": 2,
                    "attacking_mid": 3,
                    "forwards": 1,
                },
            },
            "basketball": {
                "1-3-1": {"point_guard": 1, "wings": 3, "center": 1},
                "2-3": {"guards": 2, "forwards": 3},
                "3-2": {"perimeter": 3, "post": 2},
            },
        }

        # Latest detected formations by team
        self.latest_formations = {}
        self.formation_history = defaultdict(list)

    def detect_formation(self, positions, sport, team_positions):
        """
        Detect team formation based on player positions

        Args:
            positions (list): List of player positions
            sport (str): Sport type
            team_positions (dict): Player positions by team

        Returns:
            str: Detected formation
        """
        if sport not in self.known_formations:
            return None

        # Use different methods based on sport
        if sport == "football":
            return self._detect_football_formation(positions)
        elif sport == "basketball":
            return self._detect_basketball_formation(positions)

        return None

    def _detect_football_formation(self, positions):
        """Detect football formations"""
        if len(positions) < 10:  # Need at least 10 players (excluding goalkeeper)
            return None

        # Sort positions by y-coordinate (assuming y-axis is along field length)
        positions_sorted = sorted(positions, key=lambda pos: pos[1])

        # Skip goalkeeper (assumed to be the last player in sorted list)
        field_players = positions_sorted[:-1]

        # Cluster players by their y-position to identify lines
        from sklearn.cluster import KMeans

        # Try clustering with different numbers of lines (typically 3 or 4)
        best_formation = None
        best_score = float("inf")

        for n_lines in [3, 4]:
            # Extract y-coordinates for clustering
            y_coords = np.array([pos[1] for pos in field_players]).reshape(-1, 1)

            # Apply KMeans clustering
            kmeans = KMeans(n_clusters=n_lines, random_state=0).fit(y_coords)
            labels = kmeans.labels_

            # Count players in each line
            line_counts = [
                sum(1 for label in labels if label == i) for i in range(n_lines)
            ]

            # Check if the line counts match any known formation
            for formation, structure in self.known_formations["football"].items():
                # Convert formation structure to line counts
                if n_lines == 3:
                    expected_counts = [
                        structure.get("defenders", 0),
                        structure.get("midfielders", 0)
                        + structure.get("defensive_mid", 0)
                        + structure.get("attacking_mid", 0),
                        structure.get("forwards", 0),
                    ]
                elif n_lines == 4:
                    expected_counts = [
                        structure.get("defenders", 0),
                        structure.get("defensive_mid", 0),
                        structure.get("midfielders", 0)
                        + structure.get("attacking_mid", 0),
                        structure.get("forwards", 0),
                    ]

                # Compare actual counts with expected counts
                score = sum(
                    abs(a - b)
                    for a, b in zip(sorted(line_counts), sorted(expected_counts))
                )

                if score < best_score:
                    best_score = score
                    best_formation = formation

        return best_formation

    def _detect_basketball_formation(self, positions):
        """Detect basketball formations"""
        if len(positions) < 5:  # Need all 5 players
            return None

        # For basketball, we look at the spatial arrangement
        from scipy.spatial import ConvexHull

        try:
            # Calculate convex hull to analyze team shape
            hull = ConvexHull(positions)
            hull_area = hull.volume  # In 2D, volume is area

            # Calculate distances from center
            center = np.mean(positions, axis=0)
            distances = [np.linalg.norm(np.array(pos) - center) for pos in positions]

            # Sort distances
            sorted_distances = sorted(distances)

            # Check formation patterns
            if sorted_distances[0] < 3 and all(d > 5 for d in sorted_distances[1:]):
                # One player near center, others spread out (1-3-1 or similar)
                return "1-3-1"
            elif (
                sorted_distances[0] < 3
                and sorted_distances[1] < 3
                and all(d > 5 for d in sorted_distances[2:])
            ):
                # Two players near center, three spread out (2-3)
                return "2-3"
            elif all(3 < d < 6 for d in sorted_distances):
                # All players at similar medium distance (3-2 or balanced)
                return "3-2"
        except:
            pass

        return None


class GameAnalyzer:
    def __init__(self):
        """Initialize game analyzer"""
        self.games = {}  # Dictionary to store multiple game analyses

    def add_game(self, game_id, tracking_data):
        """Add tracking data from a game"""
        self.games[game_id] = tracking_data

    def load_game_from_file(self, game_id, filepath):
        """Load tracking data from pickle file"""
        try:
            with open(filepath, "rb") as f:
                tracking_data = pickle.load(f)

            self.games[game_id] = tracking_data
            return True

        except Exception as e:
            print(f"Error loading game data: {e}")
            return False

    def load_game_from_csv(self, game_id, csv_path):
        """Load tracking data from CSV file"""
        try:
            df = pd.read_csv(csv_path)

            # Convert DataFrame to tracking data format
            players = defaultdict(list)
            player_speeds = defaultdict(list)
            player_distances = defaultdict(float)
            player_teams = {}

            # Group by player_id
            for player_id, group in df.groupby("player_id"):
                # Sort by position_index to ensure correct order
                group = group.sort_values("position_index")

                # Extract positions
                positions = [(row["x"], row["y"]) for _, row in group.iterrows()]
                players[player_id] = positions

                # Extract speeds
                speeds = [
                    row["speed"]
                    for _, row in group.iterrows()
                    if not pd.isna(row["speed"])
                ]
                player_speeds[player_id] = speeds

                # Extract team
                if "team" in group.columns:
                    team = group["team"].iloc[0]
                    if not pd.isna(team):
                        player_teams[player_id] = int(team)

                # Calculate total distance
                if len(positions) > 1:
                    for i in range(len(positions) - 1):
                        player_distances[player_id] += distance.euclidean(
                            positions[i], positions[i + 1]
                        )

            # Create tracking data dictionary
            tracking_data = {
                "player_tracks": dict(players),
                "player_speeds": dict(player_speeds),
                "player_distances": dict(player_distances),
                "player_teams": player_teams,
            }

            self.games[game_id] = tracking_data
            return True

        except Exception as e:
            print(f"Error loading game data from CSV: {e}")
            return False

    def compare_player_performance(self, player_id, metrics=None):
        """Compare player performance across multiple games"""
        if not metrics:
            metrics = ["total_distance", "avg_speed", "max_speed"]

        comparison = {}

        for game_id, game_data in self.games.items():
            # Skip if player not in this game
            if player_id not in game_data["player_tracks"]:
                continue

            player_stats = {}

            # Get total distance
            if "total_distance" in metrics and "player_distances" in game_data:
                player_stats["total_distance"] = game_data["player_distances"].get(
                    player_id, 0
                )

            # Get speed stats
            if "player_speeds" in game_data and player_id in game_data["player_speeds"]:
                speeds = game_data["player_speeds"][player_id]
                if speeds:
                    if "avg_speed" in metrics:
                        player_stats["avg_speed"] = sum(speeds) / len(speeds)
                    if "max_speed" in metrics:
                        player_stats["max_speed"] = max(speeds)

            # Get action counts if available
            if (
                "player_actions" in game_data
                and player_id in game_data["player_actions"]
            ):
                actions = game_data["player_actions"][player_id]
                action_counts = defaultdict(int)
                for _, action in actions:
                    action_counts[action] += 1

                if "action_counts" in metrics:
                    player_stats["action_counts"] = dict(action_counts)

                # Count specific actions if requested
                for metric in metrics:
                    if metric.startswith("action_"):
                        action = metric[7:]  # Extract action name
                        player_stats[metric] = action_counts.get(action, 0)

            # Zone percentages if available
            if "zone_data" in game_data and player_id in game_data["zone_data"]:
                zone_data = game_data["zone_data"][player_id]
                total_frames = sum(zone_data.values())

                if total_frames > 0 and "zone_percentages" in metrics:
                    zone_percentages = {
                        zone: (count / total_frames * 100)
                        for zone, count in zone_data.items()
                    }
                    player_stats["zone_percentages"] = zone_percentages

            # Add to comparison if we have data
            if player_stats:
                comparison[game_id] = player_stats

        return comparison

    def generate_team_heatmap(self, game_id, team=None):
        """Generate a team heatmap for a specific game"""
        if game_id not in self.games:
            return None

        game_data = self.games[game_id]

        # Use PlayerTracker's heatmap generation method
        if "player_tracks" not in game_data or "player_teams" not in game_data:
            return None

        # Create temporary PlayerTracker to generate heatmap
        temp_tracker = PlayerTracker()
        temp_tracker.players = game_data["player_tracks"]
        temp_tracker.player_teams = game_data["player_teams"]

        return temp_tracker.generate_heatmap_data(team=team)

    def analyze_possession(self, game_id):
        """Analyze possession statistics for a game"""
        if game_id not in self.games:
            return None

        game_data = self.games[game_id]

        if "possession_data" not in game_data:
            return None

        possession_data = game_data["possession_data"]

        # Count frames for each team
        team_frames = defaultdict(int)
        for frame, data in possession_data.items():
            team = data.get("team", -1)
            if team != -1:
                team_frames[team] += 1

        total_frames = sum(team_frames.values())

        if total_frames == 0:
            return None

        # Calculate possession percentages
        possession_percentages = {
            team: (frames / total_frames * 100) for team, frames in team_frames.items()
        }

        # Calculate possession by zone
        zone_possession = defaultdict(lambda: defaultdict(int))

        for frame, data in possession_data.items():
            team = data.get("team", -1)
            if team != -1:
                zone_control = data.get("zone_control", {})
                for zone, control in zone_control.items():
                    for team, count in control.items():
                        zone_possession[zone][team] += count

        # Calculate possession by time periods
        if "total_frames" in game_data and "frame_rate" in game_data:
            total_frames = game_data["total_frames"]
            frame_rate = game_data["frame_rate"]

            # Divide game into quarters
            quarter_size = total_frames // 4
            quarters = {}

            for i in range(4):
                start_frame = i * quarter_size
                end_frame = (i + 1) * quarter_size if i < 3 else total_frames

                quarter_frames = defaultdict(int)
                for frame in range(start_frame, end_frame):
                    if frame in possession_data:
                        team = possession_data[frame].get("team", -1)
                        if team != -1:
                            quarter_frames[team] += 1

                quarter_total = sum(quarter_frames.values())
                if quarter_total > 0:
                    quarters[f"Q{i+1}"] = {
                        team: (frames / quarter_total * 100)
                        for team, frames in quarter_frames.items()
                    }
        else:
            quarters = None

        return {
            "possession_percentages": possession_percentages,
            "zone_possession": {
                zone: dict(teams) for zone, teams in zone_possession.items()
            },
            "quarters": quarters,
        }

    def player_interaction_analysis(self, game_id, distance_threshold=5):
        """Analyze player interactions based on proximity"""
        if game_id not in self.games:
            return None

        game_data = self.games[game_id]

        # Use PlayerTracker's interaction analysis method
        if "player_tracks" not in game_data or "player_teams" not in game_data:
            return None

        # Create temporary PlayerTracker to analyze interactions
        temp_tracker = PlayerTracker()
        temp_tracker.players = game_data["player_tracks"]
        temp_tracker.player_teams = game_data["player_teams"]
        temp_tracker.player_positions = game_data.get(
            "frame_positions", defaultdict(list)
        )

        return temp_tracker.analyze_interactions(distance_threshold)

    def formation_analysis(self, game_id):
        """Analyze team formations over time"""
        if game_id not in self.games:
            return None

        game_data = self.games[game_id]

        # Check if we have the required data
        if "player_tracks" not in game_data or "player_teams" not in game_data:
            return None

        # Get player positions by frame
        if "frame_positions" not in game_data:
            return None

        frame_positions = game_data["frame_positions"]

        # Create formation detector
        formation_detector = FormationDetector()

        # Analyze formations at regular intervals
        interval = 30  # Every second for a 30fps video
        formations = defaultdict(list)

        for frame in sorted(frame_positions.keys()):
            if frame % interval != 0:
                continue

            # Get player positions by team
            team_positions = defaultdict(list)
            for player_id, pos in frame_positions[frame]:
                team = game_data["player_teams"].get(player_id, -1)
                if team != -1:
                    team_positions[team].append((player_id, pos))

            # Detect formation for each team
            for team, positions in team_positions.items():
                if len(positions) >= 5:  # Need at least 5 players
                    formation = formation_detector.detect_formation(
                        [pos for _, pos in positions],
                        "football",  # Assuming football
                        team_positions,
                    )

                    if formation:
                        formations[team].append((frame, formation))

        return dict(formations)

    def generate_event_timeline(self, game_id):
        """Generate timeline of significant events"""
        if game_id not in self.games:
            return None

        game_data = self.games[game_id]

        if "events" not in game_data:
            return None

        events = game_data["events"]
        frame_rate = game_data.get(
            "frame_rate", 30
        )  # Default to 30fps if not available

        # Create timeline data
        timeline = []

        for event in events:
            # Convert frame to time
            time_seconds = event["frame"] / frame_rate
            minutes = int(time_seconds // 60)
            seconds = int(time_seconds % 60)
            time_str = f"{minutes:02d}:{seconds:02d}"

            timeline.append(
                {
                    "time": time_str,
                    "time_seconds": time_seconds,
                    "action": event["action"],
                    "team": event.get("team", -1),
                    "player": event.get("player", None),
                    "position": event.get("position", None),
                }
            )

        return timeline

    def export_analytics_report(self, game_id, filename):
        """Export comprehensive analytics report as HTML"""
        if game_id not in self.games:
            return False

        game_data = self.games[game_id]

        # Create visualizer
        visualizer = Visualizer()

        # Generate various analytics
        team_heatmap_0 = self.generate_team_heatmap(game_id, team=0)
        team_heatmap_1 = self.generate_team_heatmap(game_id, team=1)
        possession_analysis = self.analyze_possession(game_id)
        interaction_analysis = self.player_interaction_analysis(game_id)
        event_timeline = self.generate_event_timeline(game_id)

        # Create figures
        figures = []

        # Team heatmaps
        if team_heatmap_0 is not None:
            fig = visualizer.plot_heatmap(
                team_heatmap_0, title="Team 1 Position Heatmap"
            )
            figures.append(("team1_heatmap", fig))

        if team_heatmap_1 is not None:
            fig = visualizer.plot_heatmap(
                team_heatmap_1, title="Team 2 Position Heatmap"
            )
            figures.append(("team2_heatmap", fig))

        # Possession chart
        if possession_analysis:
            fig = visualizer.plot_possession_chart(
                possession_analysis["possession_percentages"]
            )
            figures.append(("possession_chart", fig))

            if possession_analysis["quarters"]:
                fig = visualizer.plot_possession_by_period(
                    possession_analysis["quarters"]
                )
                figures.append(("possession_periods", fig))

        # Player interaction network
        if interaction_analysis:
            player_positions = {
                player_id: np.mean(positions, axis=0)
                for player_id, positions in game_data["player_tracks"].items()
            }

            fig = visualizer.plot_interaction_network(
                interaction_analysis["player_interactions"],
                player_positions,
                game_data["player_teams"],
                title="Player Interaction Network",
            )
            figures.append(("interaction_network", fig))

        # Event timeline
        if event_timeline:
            fig = visualizer.plot_event_timeline(event_timeline)
            figures.append(("event_timeline", fig))

        # Player stats charts for a few key players
        top_players = self._get_top_players(game_data, n=5)
        for player_id in top_players:
            fig = visualizer.plot_player_stats(player_id, game_data)
            figures.append((f"player_{player_id}_stats", fig))

        # Generate HTML report
        html_content = self._generate_html_report(game_id, game_data, figures)

        # Save report
        with open(filename, "w") as f:
            f.write(html_content)

        return True

    def _get_top_players(self, game_data, n=5):
        """Get top n players based on distance covered"""
        if (
            "player_distances" not in game_data
            or len(game_data["player_distances"]) == 0
        ):
            return []

        # Sort players by distance covered
        sorted_players = sorted(
            game_data["player_distances"].items(), key=lambda x: x[1], reverse=True
        )

        return [player_id for player_id, _ in sorted_players[:n]]

    def _generate_html_report(self, game_id, game_data, figures):
        """Generate HTML report content"""
        # Convert Plotly figures to HTML
        figure_html = {}
        for name, fig in figures:
            figure_html[name] = pio.to_html(
                fig, include_plotlyjs="cdn", full_html=False
            )

        # Basic game stats
        total_frames = game_data.get("total_frames", 0)
        frame_rate = game_data.get("frame_rate", 30)
        duration = total_frames / frame_rate if frame_rate > 0 else 0
        minutes = int(duration // 60)
        seconds = int(duration % 60)

        # Create HTML content
        html = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Sports Analytics Report - Game {game_id}</title>
            <meta charset="UTF-8">
            <meta name="viewport" content="width=device-width, initial-scale=1.0">
            <style>
                body {{ font-family: Arial, sans-serif; margin: 0; padding: 20px; color: #333; }}
                .header {{ background-color: #1a5276; color: white; padding: 20px; margin-bottom: 20px; }}
                .section {{ margin-bottom: 30px; background-color: #f9f9f9; padding: 20px; border-radius: 5px; }}
                h1, h2, h3 {{ color: #2c3e50; }}
                .stats-container {{ display: flex; flex-wrap: wrap; gap: 15px; }}
                .stat-box {{ background-color: #eaecee; padding: 15px; border-radius: 5px; min-width: 150px; flex: 1; }}
                .chart-container {{ margin-top: 20px; }}
                table {{ width: 100%; border-collapse: collapse; margin-top: 15px; }}
                th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
                th {{ background-color: #f2f2f2; }}
                tr:nth-child(even) {{ background-color: #f9f9f9; }}
            </style>
        </head>
        <body>
            <div class="header">
                <h1>Sports Analytics Report</h1>
                <h2>Game {game_id}</h2>
            </div>
            
            <div class="section">
                <h2>Game Overview</h2>
                <div class="stats-container">
                    <div class="stat-box">
                        <h3>Duration</h3>
                        <p>{minutes} minutes, {seconds} seconds</p>
                    </div>
                    <div class="stat-box">
                        <h3>Players Tracked</h3>
                        <p>{len(game_data.get("player_tracks", {}))} players</p>
                    </div>
                    <div class="stat-box">
                        <h3>Teams</h3>
                        <p>2 teams</p>
                    </div>
                    <div class="stat-box">
                        <h3>Events Detected</h3>
                        <p>{len(game_data.get("events", []))} events</p>
                    </div>
                </div>
            </div>
        """

        # Add team analysis section
        html += """
            <div class="section">
                <h2>Team Analysis</h2>
        """

        # Add possession charts if available
        if "possession_chart" in figure_html:
            html += f"""
                <div class="chart-container">
                    <h3>Ball Possession</h3>
                    {figure_html["possession_chart"]}
                </div>
            """

        if "possession_periods" in figure_html:
            html += f"""
                <div class="chart-container">
                    <h3>Possession by Period</h3>
                    {figure_html["possession_periods"]}
                </div>
            """

        # Add team heatmaps
        if "team1_heatmap" in figure_html:
            html += f"""
                <div class="chart-container">
                    <h3>Team 1 Position Heatmap</h3>
                    {figure_html["team1_heatmap"]}
                </div>
            """

        if "team2_heatmap" in figure_html:
            html += f"""
                <div class="chart-container">
                    <h3>Team 2 Position Heatmap</h3>
                    {figure_html["team2_heatmap"]}
                </div>
            """

        html += """
            </div>
        """

        # Add player analysis section
        html += """
            <div class="section">
                <h2>Player Analysis</h2>
        """

        # Add player stats for top players
        for name, fig_html in figure_html.items():
            if name.startswith("player_") and name.endswith("_stats"):
                player_id = name.split("_")[1]
                html += f"""
                    <div class="chart-container">
                        <h3>Player {player_id} Statistics</h3>
                        {fig_html}
                    </div>
                """

        # Add interaction network
        if "interaction_network" in figure_html:
            html += f"""
                <div class="chart-container">
                    <h3>Player Interaction Network</h3>
                    {figure_html["interaction_network"]}
                </div>
            """

        html += """
            </div>
        """

        # Add events section
        html += """
            <div class="section">
                <h2>Key Events</h2>
        """

        # Add event timeline
        if "event_timeline" in figure_html:
            html += f"""
                <div class="chart-container">
                    <h3>Event Timeline</h3>
                    {figure_html["event_timeline"]}
                </div>
            """

        # Add event table
        events = game_data.get("events", [])
        if events:
            frame_rate = game_data.get("frame_rate", 30)

            html += """
                <h3>Event List</h3>
                <table>
                    <tr>
                        <th>Time</th>
                        <th>Action</th>
                        <th>Team</th>
                        <th>Player</th>
                    </tr>
            """

            for event in sorted(events, key=lambda e: e.get("frame", 0)):
                # Convert frame to time
                time_seconds = event.get("frame", 0) / frame_rate
                minutes = int(time_seconds // 60)
                seconds = int(time_seconds % 60)
                time_str = f"{minutes:02d}:{seconds:02d}"

                team = event.get("team", -1)
                team_str = f"Team {team+1}" if team != -1 else "N/A"

                player = event.get("player", None)
                player_str = f"Player {player}" if player is not None else "N/A"

                html += f"""
                    <tr>
                        <td>{time_str}</td>
                        <td>{event.get("action", "Unknown")}</td>
                        <td>{team_str}</td>
                        <td>{player_str}</td>
                    </tr>
                """

            html += """
                </table>
            """

        html += """
            </div>
        """

        # Add footer and close HTML
        html += f"""
            <div class="section">
                <h2>Analysis Information</h2>
                <p>Analysis generated on: {datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")}</p>
                <p>Total processing time: {game_data.get("processing_duration", 0):.2f} seconds</p>
            </div>
            
            <div class="footer" style="text-align: center; margin-top: 30px; padding: 20px; color: #7f8c8d;">
                <p>AI-Powered Sports Analytics & Player Tracking System</p>
            </div>
        </body>
        </html>
        """

        return html


class Visualizer:
    def __init__(self):
        """Initialize visualizer"""
        self.colormap = CONFIG["visualization_settings"]["default_colormap"]
        self.field_overlay_opacity = CONFIG["visualization_settings"][
            "field_overlay_opacity"
        ]
        self.trajectory_line_width = CONFIG["visualization_settings"][
            "trajectory_line_width"
        ]
        self.marker_size = CONFIG["visualization_settings"]["marker_size"]
        self.font_size = CONFIG["visualization_settings"]["font_size"]

    def plot_player_trajectory(
        self,
        player_tracks,
        player_id,
        field_width=105,
        field_height=68,
        sport="football",
        title=None,
        show_field=True,
    ):
        """
        Plot the trajectory of a specific player

        Args:
            player_tracks (dict): Dictionary of player tracks
            player_id (int): ID of player to plot
            field_width (float): Width of the field in meters
            field_height (float): Height of the field in meters
            sport (str): Sport type for field markings
            title (str): Plot title
            show_field (bool): Whether to show field markings

        Returns:
            plotly.graph_objects.Figure: Trajectory plot figure
        """
        if player_id not in player_tracks:
            return None

        positions = player_tracks[player_id]
        x_coords, y_coords = zip(*positions)

        # Create figure
        fig = go.Figure()

        if show_field:
            # Add field background based on sport
            if sport == "football":
                # Football field
                fig.add_shape(
                    type="rect",
                    x0=0,
                    y0=0,
                    x1=field_width,
                    y1=field_height,
                    line=dict(color="darkgreen", width=2),
                    fillcolor="lightgreen",
                    opacity=self.field_overlay_opacity,
                )

                # Center circle
                fig.add_shape(
                    type="circle",
                    x0=field_width / 2 - 9.15,
                    y0=field_height / 2 - 9.15,
                    x1=field_width / 2 + 9.15,
                    y1=field_height / 2 + 9.15,
                    line=dict(color="white", width=2),
                    fillcolor=None,
                )

                # Center line
                fig.add_shape(
                    type="line",
                    x0=field_width / 2,
                    y0=0,
                    x1=field_width / 2,
                    y1=field_height,
                    line=dict(color="white", width=2),
                )

                # Penalty areas
                fig.add_shape(
                    type="rect",
                    x0=0,
                    y0=(field_height - 40.3) / 2,
                    x1=16.5,
                    y1=(field_height + 40.3) / 2,
                    line=dict(color="white", width=2),
                    fillcolor=None,
                )

                fig.add_shape(
                    type="rect",
                    x0=field_width - 16.5,
                    y0=(field_height - 40.3) / 2,
                    x1=field_width,
                    y1=(field_height + 40.3) / 2,
                    line=dict(color="white", width=2),
                    fillcolor=None,
                )

                # Goal areas
                fig.add_shape(
                    type="rect",
                    x0=0,
                    y0=(field_height - 18.3) / 2,
                    x1=5.5,
                    y1=(field_height + 18.3) / 2,
                    line=dict(color="white", width=2),
                    fillcolor=None,
                )

                fig.add_shape(
                    type="rect",
                    x0=field_width - 5.5,
                    y0=(field_height - 18.3) / 2,
                    x1=field_width,
                    y1=(field_height + 18.3) / 2,
                    line=dict(color="white", width=2),
                    fillcolor=None,
                )

            elif sport == "basketball":
                # Basketball court
                fig.add_shape(
                    type="rect",
                    x0=0,
                    y0=0,
                    x1=field_width,
                    y1=field_height,
                    line=dict(color="brown", width=2),
                    fillcolor="bisque",
                    opacity=self.field_overlay_opacity,
                )

                # Center circle
                fig.add_shape(
                    type="circle",
                    x0=field_width / 2 - 1.8,
                    y0=field_height / 2 - 1.8,
                    x1=field_width / 2 + 1.8,
                    y1=field_height / 2 + 1.8,
                    line=dict(color="brown", width=2),
                    fillcolor=None,
                )

                # Half-court line
                fig.add_shape(
                    type="line",
                    x0=field_width / 2,
                    y0=0,
                    x1=field_width / 2,
                    y1=field_height,
                    line=dict(color="brown", width=2),
                )

                # Three-point lines
                radius = 6.75
                fig.add_shape(
                    type="circle",
                    x0=0 - radius,
                    y0=field_height / 2 - radius,
                    x1=0 + radius,
                    y1=field_height / 2 + radius,
                    line=dict(color="brown", width=2),
                    fillcolor=None,
                )

                fig.add_shape(
                    type="circle",
                    x0=field_width - radius,
                    y0=field_height / 2 - radius,
                    x1=field_width + radius,
                    y1=field_height / 2 + radius,
                    line=dict(color="brown", width=2),
                    fillcolor=None,
                )

        # Add player trajectory
        color_scale = px.colors.sequential.Viridis
        n_points = len(x_coords)
        colors = [
            color_scale[min(int(i / n_points * len(color_scale)), len(color_scale) - 1)]
            for i in range(n_points)
        ]

        fig.add_trace(
            go.Scatter(
                x=x_coords,
                y=y_coords,
                mode="lines",
                line=dict(
                    color="blue",
                    width=self.trajectory_line_width,
                    colorscale=self.colormap,
                ),
                name=f"Player {player_id}",
            )
        )

        # Add points with color gradient to show time progression
        fig.add_trace(
            go.Scatter(
                x=x_coords,
                y=y_coords,
                mode="markers",
                marker=dict(
                    size=self.marker_size,
                    color=list(range(len(x_coords))),
                    colorscale=self.colormap,
                    showscale=True,
                    colorbar=dict(title="Time progression"),
                ),
                name=f"Player {player_id} positions",
            )
        )

        # Add start and end points
        fig.add_trace(
            go.Scatter(
                x=[x_coords[0]],
                y=[y_coords[0]],
                mode="markers",
                name="Start",
                marker=dict(
                    color="green", size=self.marker_size * 1.5, symbol="circle"
                ),
            )
        )

        fig.add_trace(
            go.Scatter(
                x=[x_coords[-1]],
                y=[y_coords[-1]],
                mode="markers",
                name="End",
                marker=dict(color="red", size=self.marker_size * 1.5, symbol="circle"),
            )
        )

        # Update layout
        fig.update_layout(
            title=title or f"Player {player_id} Trajectory",
            xaxis_title="X Position (meters)",
            yaxis_title="Y Position (meters)",
            legend=dict(x=0, y=1),
            width=800,
            height=600,
            xaxis=dict(range=[0, field_width]),
            yaxis=dict(range=[0, field_height]),
            font=dict(size=self.font_size),
        )

        return fig

    def plot_speed_profile(
        self,
        player_speeds,
        player_id,
        frame_rate=25,
        title=None,
        highlight_max=True,
        show_threshold=True,
    ):
        """
        Plot the speed profile of a specific player

        Args:
            player_speeds (dict): Dictionary of player speeds
            player_id (int): ID of player to plot
            frame_rate (float): Video frame rate
            title (str): Plot title
            highlight_max (bool): Whether to highlight maximum speed
            show_threshold (bool): Whether to show speed thresholds

        Returns:
            plotly.graph_objects.Figure: Speed profile figure
        """
        if player_id not in player_speeds:
            return None

        speeds = player_speeds[player_id]

        # Create time axis (in seconds)
        time_points = [i / frame_rate for i in range(len(speeds))]

        # Create figure
        fig = go.Figure()

        # Add speed trace
        fig.add_trace(
            go.Scatter(
                x=time_points,
                y=speeds,
                mode="lines",
                name=f"Player {player_id} Speed",
                line=dict(color="blue", width=2),
            )
        )

        # Add average speed line
        avg_speed = sum(speeds) / len(speeds)
        fig.add_trace(
            go.Scatter(
                x=[time_points[0], time_points[-1]],
                y=[avg_speed, avg_speed],
                mode="lines",
                name="Average Speed",
                line=dict(color="red", width=2, dash="dash"),
            )
        )

        # Highlight maximum speed
        if highlight_max:
            max_speed = max(speeds)
            max_index = speeds.index(max_speed)
            max_time = time_points[max_index]

            fig.add_trace(
                go.Scatter(
                    x=[max_time],
                    y=[max_speed],
                    mode="markers",
                    name="Maximum Speed",
                    marker=dict(color="red", size=self.marker_size * 1.5),
                )
            )

            # Add annotation
            fig.add_annotation(
                x=max_time,
                y=max_speed,
                text=f"Max: {max_speed:.2f} m/s",
                showarrow=True,
                arrowhead=1,
                ax=0,
                ay=-40,
            )

        # Add speed thresholds
        if show_threshold:
            # Walking, jogging, running thresholds
            thresholds = [
                (2.0, "Walking"),
                (4.0, "Jogging"),
                (7.0, "Running"),
                (9.0, "Sprinting"),
            ]
            colors = ["green", "orange", "red", "purple"]

            for i, (threshold, label) in enumerate(thresholds):
                if i < len(thresholds) - 1:
                    next_threshold = thresholds[i + 1][0]
                    # Add colored background for range
                    fig.add_shape(
                        type="rect",
                        x0=time_points[0],
                        x1=time_points[-1],
                        y0=threshold,
                        y1=next_threshold,
                        fillcolor=colors[i],
                        opacity=0.1,
                        layer="below",
                        line_width=0,
                    )
                else:
                    # Last threshold - color above
                    fig.add_shape(
                        type="rect",
                        x0=time_points[0],
                        x1=time_points[-1],
                        y0=threshold,
                        y1=max(speeds) * 1.1,
                        fillcolor=colors[i],
                        opacity=0.1,
                        layer="below",
                        line_width=0,
                    )

                # Add threshold line
                fig.add_trace(
                    go.Scatter(
                        x=[time_points[0], time_points[-1]],
                        y=[threshold, threshold],
                        mode="lines",
                        name=label,
                        line=dict(color=colors[i], width=1, dash="dot"),
                    )
                )

        # Update layout
        fig.update_layout(
            title=title or f"Player {player_id} Speed Profile",
            xaxis_title="Time (seconds)",
            yaxis_title="Speed (m/s)",
            legend=dict(x=0, y=1),
            width=800,
            height=400,
            font=dict(size=self.font_size),
        )

        return fig

    def plot_heatmap(
        self,
        heatmap_data,
        title=None,
        field_width=105,
        field_height=68,
        sport="football",
    ):
        """
        Plot a heatmap of player positions

        Args:
            heatmap_data (numpy.ndarray): Heatmap data
            title (str): Plot title
            field_width (float): Width of the field in meters
            field_height (float): Height of the field in meters
            sport (str): Sport type for field markings

        Returns:
            plotly.graph_objects.Figure: Heatmap figure
        """
        # Create figure
        fig = go.Figure()

        # Add field background based on sport
        x = np.linspace(0, field_width, heatmap_data.shape[1])
        y = np.linspace(0, field_height, heatmap_data.shape[0])

        # Add heatmap
        fig.add_trace(
            go.Heatmap(
                z=heatmap_data,
                x=x,
                y=y,
                colorscale=self.colormap,
                showscale=True,
                colorbar=dict(title="Density"),
            )
        )

        # Add field markings
        if sport == "football":
            # Add center circle
            circle_x = [
                field_width / 2 + 9.15 * np.cos(t)
                for t in np.linspace(0, 2 * np.pi, 100)
            ]
            circle_y = [
                field_height / 2 + 9.15 * np.sin(t)
                for t in np.linspace(0, 2 * np.pi, 100)
            ]

            fig.add_trace(
                go.Scatter(
                    x=circle_x,
                    y=circle_y,
                    mode="lines",
                    line=dict(color="white", width=2),
                    showlegend=False,
                )
            )

            # Add center line
            fig.add_trace(
                go.Scatter(
                    x=[field_width / 2, field_width / 2],
                    y=[0, field_height],
                    mode="lines",
                    line=dict(color="white", width=2),
                    showlegend=False,
                )
            )

            # Add penalty areas
            fig.add_trace(
                go.Scatter(
                    x=[0, 16.5, 16.5, 0],
                    y=[
                        (field_height - 40.3) / 2,
                        (field_height - 40.3) / 2,
                        (field_height + 40.3) / 2,
                        (field_height + 40.3) / 2,
                    ],
                    mode="lines",
                    line=dict(color="white", width=2),
                    fill=None,
                    showlegend=False,
                )
            )

            fig.add_trace(
                go.Scatter(
                    x=[
                        field_width,
                        field_width - 16.5,
                        field_width - 16.5,
                        field_width,
                    ],
                    y=[
                        (field_height - 40.3) / 2,
                        (field_height - 40.3) / 2,
                        (field_height + 40.3) / 2,
                        (field_height + 40.3) / 2,
                    ],
                    mode="lines",
                    line=dict(color="white", width=2),
                    fill=None,
                    showlegend=False,
                )
            )

        elif sport == "basketball":
            # Add center circle
            circle_x = [
                field_width / 2 + 1.8 * np.cos(t)
                for t in np.linspace(0, 2 * np.pi, 100)
            ]
            circle_y = [
                field_height / 2 + 1.8 * np.sin(t)
                for t in np.linspace(0, 2 * np.pi, 100)
            ]

            fig.add_trace(
                go.Scatter(
                    x=circle_x,
                    y=circle_y,
                    mode="lines",
                    line=dict(color="white", width=2),
                    showlegend=False,
                )
            )

            # Add half-court line
            fig.add_trace(
                go.Scatter(
                    x=[field_width / 2, field_width / 2],
                    y=[0, field_height],
                    mode="lines",
                    line=dict(color="white", width=2),
                    showlegend=False,
                )
            )

        # Update layout
        fig.update_layout(
            title=title or "Position Heatmap",
            width=800,
            height=int(800 * (field_height / field_width)),
            xaxis=dict(title="X Position (meters)", range=[0, field_width]),
            yaxis=dict(title="Y Position (meters)", range=[0, field_height]),
            font=dict(size=self.font_size),
        )

        return fig

    def plot_player_stats(self, player_id, game_data):
        """
        Plot comprehensive player statistics

        Args:
            player_id (int): Player ID
            game_data (dict): Game data dictionary

        Returns:
            plotly.graph_objects.Figure: Player stats figure
        """
        # Extract player data
        player_speeds = game_data.get("player_speeds", {}).get(player_id, [])
        total_distance = game_data.get("player_distances", {}).get(player_id, 0)
        player_actions = game_data.get("player_actions", {}).get(player_id, [])
        team = game_data.get("player_teams", {}).get(player_id, -1)

        # Calculate stats
        avg_speed = sum(player_speeds) / len(player_speeds) if player_speeds else 0
        max_speed = max(player_speeds) if player_speeds else 0

        # Count actions
        action_counts = defaultdict(int)
        for _, action in player_actions:
            action_counts[action] += 1

        # Create subplot figure
        fig = make_subplots(
            rows=2,
            cols=2,
            specs=[
                [{"type": "indicator"}, {"type": "indicator"}],
                [{"type": "bar"}, {"type": "pie"}],
            ],
            subplot_titles=(
                "Total Distance",
                "Max Speed",
                "Actions",
                "Zone Distribution",
            ),
        )

        # Add distance indicator
        fig.add_trace(
            go.Indicator(
                mode="number+gauge",
                value=total_distance,
                number={"suffix": " m", "font": {"size": 24}},
                gauge={
                    "axis": {"range": [0, 12000]},
                    "bar": {"color": "blue"},
                    "steps": [
                        {"range": [0, 4000], "color": "lightgray"},
                        {"range": [4000, 8000], "color": "gray"},
                        {"range": [8000, 12000], "color": "darkgray"},
                    ],
                },
                title={"text": "Total Distance"},
            ),
            row=1,
            col=1,
        )

        # Add speed indicator
        fig.add_trace(
            go.Indicator(
                mode="number+gauge",
                value=max_speed,
                number={"suffix": " m/s", "font": {"size": 24}},
                gauge={
                    "axis": {"range": [0, 10]},
                    "bar": {"color": "red"},
                    "steps": [
                        {"range": [0, 4], "color": "lightgreen"},
                        {"range": [4, 7], "color": "yellow"},
                        {"range": [7, 10], "color": "orange"},
                    ],
                },
                title={"text": "Max Speed"},
            ),
            row=1,
            col=2,
        )

        # Add actions bar chart
        if action_counts:
            actions = list(action_counts.keys())
            counts = list(action_counts.values())

            fig.add_trace(
                go.Bar(x=actions, y=counts, marker_color="rgb(26, 118, 255)"),
                row=2,
                col=1,
            )

        # Add zone distribution if available
        if "zone_data" in game_data and player_id in game_data["zone_data"]:
            zone_data = game_data["zone_data"][player_id]
            zones = list(zone_data.keys())
            zone_counts = list(zone_data.values())

            fig.add_trace(
                go.Pie(labels=zones, values=zone_counts, hole=0.3), row=2, col=2
            )

        # Update layout
        fig.update_layout(
            title=f"Player {player_id} Statistics"
            + (f" (Team {team+1})" if team != -1 else ""),
            height=700,
            width=800,
            showlegend=False,
            font=dict(size=self.font_size),
        )

        return fig

    def plot_possession_chart(self, possession_percentages):
        """
        Plot possession percentages

        Args:
            possession_percentages (dict): Team possession percentages

        Returns:
            plotly.graph_objects.Figure: Possession chart figure
        """
        # Create figure
        fig = go.Figure()

        # Get teams and percentages
        teams = []
        percentages = []
        colors = []

        for team, percentage in possession_percentages.items():
            teams.append(f"Team {team+1}")
            percentages.append(percentage)
            colors.append("red" if team == 0 else "blue")

        # Add pie chart
        fig.add_trace(
            go.Pie(
                labels=teams, values=percentages, hole=0.4, marker=dict(colors=colors)
            )
        )

        # Update layout
        fig.update_layout(
            title="Ball Possession",
            height=500,
            width=600,
            font=dict(size=self.font_size),
        )

        return fig

    def plot_possession_by_period(self, quarters):
        """
        Plot possession by game period

        Args:
            quarters (dict): Possession data by quarter

        Returns:
            plotly.graph_objects.Figure: Possession by period figure
        """
        # Create figure
        fig = go.Figure()

        # Extract data for each team
        periods = list(quarters.keys())
        team_0_data = [quarters[period].get(0, 0) for period in periods]
        team_1_data = [quarters[period].get(1, 0) for period in periods]

        # Add stacked bars
        fig.add_trace(
            go.Bar(x=periods, y=team_0_data, name="Team 1", marker_color="red")
        )

        fig.add_trace(
            go.Bar(x=periods, y=team_1_data, name="Team 2", marker_color="blue")
        )

        # Update layout
        fig.update_layout(
            title="Possession by Period",
            xaxis_title="Game Period",
            yaxis_title="Possession (%)",
            barmode="group",
            height=500,
            width=600,
            font=dict(size=self.font_size),
        )

        return fig

    def plot_interaction_network(
        self, player_interactions, player_positions, player_teams, title=None
    ):
        """
        Plot player interaction network

        Args:
            player_interactions (dict): Player interaction counts
            player_positions (dict): Average player positions
            player_teams (dict): Player team assignments
            title (str): Plot title

        Returns:
            plotly.graph_objects.Figure: Interaction network figure
        """
        # Create figure
        fig = go.Figure()

        # Prepare node data
        nodes = {}
        for player_id, position in player_positions.items():
            team = player_teams.get(player_id, -1)
            nodes[player_id] = {
                "position": position,
                "team": team,
                "color": "red" if team == 0 else "blue" if team == 1 else "gray",
            }

        # Calculate edge sizes based on interaction counts
        max_count = max(player_interactions.values()) if player_interactions else 1

        # Add edges (interactions)
        for (p1, p2), count in player_interactions.items():
            if p1 in nodes and p2 in nodes:
                pos1 = nodes[p1]["position"]
                pos2 = nodes[p2]["position"]

                # Line width based on interaction count
                width = max(1, (count / max_count) * 5)

                # Line color based on whether players are from same team
                same_team = (
                    nodes[p1]["team"] == nodes[p2]["team"] and nodes[p1]["team"] != -1
                )
                color = "rgba(0,100,0,0.3)" if same_team else "rgba(100,0,0,0.3)"

                fig.add_trace(
                    go.Scatter(
                        x=[pos1[0], pos2[0]],
                        y=[pos1[1], pos2[1]],
                        mode="lines",
                        line=dict(width=width, color=color),
                        showlegend=False,
                    )
                )

        # Add nodes (players)
        for team in [-1, 0, 1]:  # Draw in layers: unknown team, team 0, team 1
            team_nodes = [p for p, data in nodes.items() if data["team"] == team]

            x_coords = [nodes[p]["position"][0] for p in team_nodes]
            y_coords = [nodes[p]["position"][1] for p in team_nodes]
            colors = [nodes[p]["color"] for p in team_nodes]
            labels = [f"P{p}" for p in team_nodes]

            fig.add_trace(
                go.Scatter(
                    x=x_coords,
                    y=y_coords,
                    mode="markers+text",
                    text=labels,
                    textposition="top center",
                    marker=dict(size=15, color=colors),
                    name=f"Team {team+1}" if team != -1 else "Unknown",
                    hoverinfo="text",
                    hovertext=[f"Player {p}" for p in team_nodes],
                )
            )

        # Update layout
        fig.update_layout(
            title=title or "Player Interaction Network",
            height=600,
            width=800,
            showlegend=True,
            font=dict(size=self.font_size),
        )

        return fig

    def plot_event_timeline(self, event_timeline):
        """
        Plot event timeline

        Args:
            event_timeline (list): List of event dictionaries

        Returns:
            plotly.graph_objects.Figure: Event timeline figure
        """
        # Create figure
        fig = go.Figure()

        # Group events by action
        events_by_action = defaultdict(list)
        for event in event_timeline:
            action = event["action"]
            events_by_action[action].append(event)

        # Add a line for each action type
        colors = px.colors.qualitative.Plotly
        color_idx = 0

        for action, events in events_by_action.items():
            # Sort events by time
            events = sorted(events, key=lambda e: e["time_seconds"])

            # Extract data
            times = [e["time_seconds"] for e in events]
            teams = [e.get("team", -1) for e in events]
            players = [e.get("player", None) for e in events]

            # Create hover text
            hover_texts = []
            for i in range(len(events)):
                team_str = f"Team {teams[i]+1}" if teams[i] != -1 else "Unknown"
                player_str = (
                    f"Player {players[i]}" if players[i] is not None else "Unknown"
                )
                hover_texts.append(
                    f"Action: {action}<br>Time: {events[i]['time']}<br>Team: {team_str}<br>Player: {player_str}"
                )

            # Add scatter plot for this action
            fig.add_trace(
                go.Scatter(
                    x=times,
                    y=[action] * len(times),
                    mode="markers",
                    name=action,
                    marker=dict(
                        symbol="circle",
                        size=12,
                        color=colors[color_idx % len(colors)],
                        line=dict(width=1, color="DarkSlateGrey"),
                    ),
                    text=hover_texts,
                    hoverinfo="text",
                )
            )

            color_idx += 1

        # Update layout
        fig.update_layout(
            title="Event Timeline",
            xaxis_title="Time (seconds)",
            yaxis_title="Event Type",
            height=500,
            width=800,
            font=dict(size=self.font_size),
        )

        return fig

    def plot_performance_comparison(self, comparison_data, metrics=None, title=None):
        """
        Plot performance comparison across multiple games

        Args:
            comparison_data (dict): Comparison data
            metrics (list): List of metrics to compare
            title (str): Plot title

        Returns:
            plotly.graph_objects.Figure: Performance comparison figure
        """
        if not comparison_data:
            return None

        if not metrics:
            metrics = list(next(iter(comparison_data.values())).keys())

        # Filter metrics that are dictionaries (like action_counts)
        simple_metrics = [
            m
            for m in metrics
            if all(
                not isinstance(game_data.get(m), dict)
                for game_data in comparison_data.values()
            )
        ]

        # Create figure
        fig = go.Figure()

        # Add bars for each metric
        for metric in simple_metrics:
            metric_values = [
                game_data.get(metric, 0) for game_data in comparison_data.values()
            ]
            game_ids = list(comparison_data.keys())

            fig.add_trace(go.Bar(x=game_ids, y=metric_values, name=metric))

        # Update layout
        fig.update_layout(
            title=title or "Performance Comparison Across Games",
            xaxis_title="Game ID",
            yaxis_title="Value",
            barmode="group",
            width=800,
            height=500,
            font=dict(size=self.font_size),
        )

        return fig

    def create_analysis_report(self, game_data, filename, sport="football"):
        """
        Create comprehensive analysis report

        Args:
            game_data (dict): Game data dictionary
            filename (str): Output filename
            sport (str): Sport type

        Returns:
            bool: Success status
        """
        try:
            # Create temporary PlayerTracker to handle analysis functions
            tracker = PlayerTracker(sport=sport)
            tracker.players = game_data.get("player_tracks", {})
            tracker.player_speeds = game_data.get("player_speeds", {})
            tracker.player_distances = game_data.get("player_distances", {})
            tracker.player_actions = game_data.get("player_actions", {})
            tracker.player_teams = game_data.get("player_teams", {})
            tracker.events = game_data.get("events", [])
            tracker.possession_data = game_data.get("possession_data", {})
            tracker.zone_data = game_data.get("zone_data", {})

            # Create GameAnalyzer
            analyzer = GameAnalyzer()
            analyzer.games["current_game"] = game_data

            # Generate analysis
            possession = analyzer.analyze_possession("current_game")
            interactions = analyzer.player_interaction_analysis("current_game")
            events = analyzer.generate_event_timeline("current_game")

            # Generate team heatmaps
            team0_heatmap = tracker.generate_heatmap_data(team=0)
            team1_heatmap = tracker.generate_heatmap_data(team=1)

            # Create report HTML
            html = (
                """
            <!DOCTYPE html>
            <html>
            <head>
                <title>Sports Analytics Report</title>
                <meta charset="utf-8">
                <meta name="viewport" content="width=device-width, initial-scale=1">
                <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
                <style>
                    body { font-family: Arial, sans-serif; margin: 0; padding: 20px; background-color: #f5f5f5; }
                    .container { max-width: 1200px; margin: 0 auto; background-color: white; padding: 20px; box-shadow: 0 0 10px rgba(0,0,0,0.1); }
                    .header { text-align: center; margin-bottom: 30px; padding-bottom: 20px; border-bottom: 1px solid #eee; }
                    .section { margin-bottom: 40px; }
                    .row { display: flex; flex-wrap: wrap; margin: 0 -15px; }
                    .col { flex: 1; padding: 0 15px; min-width: 300px; }
                    h1, h2, h3 { color: #333; }
                    .stats-box { background-color: #f9f9f9; padding: 15px; border-radius: 5px; margin-bottom: 20px; }
                    .plot-container { margin-bottom: 30px; }
                    table { width: 100%; border-collapse: collapse; }
                    th, td { padding: 8px; text-align: left; border-bottom: 1px solid #ddd; }
                    th { background-color: #f2f2f2; }
                </style>
            </head>
            <body>
                <div class="container">
                    <div class="header">
                        <h1>Sports Analytics Report</h1>
                        <p>Generated on: """
                + datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                + """</p>
                    </div>
            """
            )

            # Team Analysis Section
            html += """
                    <div class="section">
                        <h2>Team Analysis</h2>
                        <div class="row">
                            <div class="col">
            """

            # Team 1 stats
            team0_stats = tracker.get_team_stats(0)
            if team0_stats:
                html += (
                    """
                                <div class="stats-box">
                                    <h3>Team 1 Stats</h3>
                                    <table>
                                        <tr><th>Stat</th><th>Value</th></tr>
                                        <tr><td>Players</td><td>"""
                    + str(team0_stats["player_count"])
                    + """</td></tr>
                                        <tr><td>Total Distance</td><td>"""
                    + f"{team0_stats['total_distance']:.2f} m"
                    + """</td></tr>
                                        <tr><td>Avg Speed</td><td>"""
                    + f"{team0_stats['avg_speed']:.2f} m/s"
                    + """</td></tr>
                                        <tr><td>Max Speed</td><td>"""
                    + f"{team0_stats['max_speed']:.2f} m/s"
                    + """</td></tr>
                                    </table>
                                </div>
                """
                )

            if possession:
                team0_possession = possession["possession_percentages"].get(0, 0)
                html += (
                    """
                                <div class="stats-box">
                                    <h3>Team 1 Possession</h3>
                                    <div style="font-size: 24px; text-align: center; margin: 15px 0;">
                                        """
                    + f"{team0_possession:.1f}%"
                    + """
                                    </div>
                                </div>
                """
                )

            html += """
                            </div>
                            <div class="col">
            """

            # Team 2 stats
            team1_stats = tracker.get_team_stats(1)
            if team1_stats:
                html += (
                    """
                                <div class="stats-box">
                                    <h3>Team 2 Stats</h3>
                                    <table>
                                        <tr><th>Stat</th><th>Value</th></tr>
                                        <tr><td>Players</td><td>"""
                    + str(team1_stats["player_count"])
                    + """</td></tr>
                                        <tr><td>Total Distance</td><td>"""
                    + f"{team1_stats['total_distance']:.2f} m"
                    + """</td></tr>
                                        <tr><td>Avg Speed</td><td>"""
                    + f"{team1_stats['avg_speed']:.2f} m/s"
                    + """</td></tr>
                                        <tr><td>Max Speed</td><td>"""
                    + f"{team1_stats['max_speed']:.2f} m/s"
                    + """</td></tr>
                                    </table>
                                </div>
                """
                )

            if possession:
                team1_possession = possession["possession_percentages"].get(1, 0)
                html += (
                    """
                                <div class="stats-box">
                                    <h3>Team 2 Possession</h3>
                                    <div style="font-size: 24px; text-align: center; margin: 15px 0;">
                                        """
                    + f"{team1_possession:.1f}%"
                    + """
                                    </div>
                                </div>
                """
                )

            html += """
                            </div>
                        </div>
            """

            # Team heatmaps
            html += """
                        <div class="row">
                            <div class="col">
                                <div class="plot-container">
                                    <h3>Team 1 Heatmap</h3>
                                    <div id="team1-heatmap"></div>
                                </div>
                            </div>
                            <div class="col">
                                <div class="plot-container">
                                    <h3>Team 2 Heatmap</h3>
                                    <div id="team2-heatmap"></div>
                                </div>
                            </div>
                        </div>
                    </div>
            """

            # Player Analysis Section
            html += """
                    <div class="section">
                        <h2>Player Analysis</h2>
                        <div class="row">
            """

            # Top players by distance
            top_players = sorted(
                game_data.get("player_distances", {}).items(),
                key=lambda x: x[1],
                reverse=True,
            )[:5]

            for player_id, _ in top_players:
                player_stats = tracker.get_player_stats(player_id)
                if player_stats:
                    team = player_stats["team"]
                    team_str = f"Team {team+1}" if team != -1 else "Unknown"

                    html += (
                        """
                            <div class="col">
                                <div class="stats-box">
                                    <h3>Player """
                        + str(player_id)
                        + """ ("""
                        + team_str
                        + """)</h3>
                                    <table>
                                        <tr><th>Stat</th><th>Value</th></tr>
                                        <tr><td>Distance</td><td>"""
                        + f"{player_stats['total_distance']:.2f} m"
                        + """</td></tr>
                                        <tr><td>Avg Speed</td><td>"""
                        + f"{player_stats['avg_speed']:.2f} m/s"
                        + """</td></tr>
                                        <tr><td>Max Speed</td><td>"""
                        + f"{player_stats['max_speed']:.2f} m/s"
                        + """</td></tr>
                                    </table>
                                    <div style="margin-top: 10px;">
                                        <h4>Actions</h4>
                                        <ul>
                    """
                    )

                    for action, count in player_stats["action_counts"].items():
                        html += (
                            """
                                            <li>"""
                            + f"{action}: {count}"
                            + """</li>
                        """
                        )

                    html += """
                                        </ul>
                                    </div>
                                </div>
                            </div>
                    """

            html += """
                        </div>
                        
                        <div class="plot-container">
                            <h3>Player Interaction Network</h3>
                            <div id="interaction-network"></div>
                        </div>
                    </div>
            """

            # Events Analysis Section
            if events:
                html += """
                    <div class="section">
                        <h2>Events Analysis</h2>
                        <div class="plot-container">
                            <h3>Event Timeline</h3>
                            <div id="event-timeline"></div>
                        </div>
                        
                        <h3>Key Events</h3>
                        <table>
                            <tr>
                                <th>Time</th>
                                <th>Action</th>
                                <th>Team</th>
                                <th>Player</th>
                            </tr>
                """

                for event in sorted(events, key=lambda e: e["time_seconds"]):
                    team_str = (
                        f"Team {event['team']+1}"
                        if event.get("team", -1) != -1
                        else "Unknown"
                    )
                    player_str = (
                        f"Player {event['player']}"
                        if event.get("player")
                        else "Unknown"
                    )

                    html += (
                        """
                            <tr>
                                <td>"""
                        + event["time"]
                        + """</td>
                                <td>"""
                        + event["action"]
                        + """</td>
                                <td>"""
                        + team_str
                        + """</td>
                                <td>"""
                        + player_str
                        + """</td>
                            </tr>
                    """
                    )

                html += """
                        </table>
                    </div>
                """

            # JavaScript for Plotly figures
            html += """
                </div>
                
                <script>
            """

            # Add team heatmap plots
            if team0_heatmap is not None:
                fig = self.plot_heatmap(
                    team0_heatmap, title="Team 1 Position Heatmap", sport=sport
                )
                plot_json = json.dumps(fig.to_dict())
                html += f"Plotly.newPlot('team1-heatmap', {plot_json}.data, {plot_json}.layout);\n"

            if team1_heatmap is not None:
                fig = self.plot_heatmap(
                    team1_heatmap, title="Team 2 Position Heatmap", sport=sport
                )
                plot_json = json.dumps(fig.to_dict())
                html += f"Plotly.newPlot('team2-heatmap', {plot_json}.data, {plot_json}.layout);\n"

            # Add interaction network
            if interactions:
                player_positions = {
                    player_id: np.mean(positions, axis=0)
                    for player_id, positions in game_data.get(
                        "player_tracks", {}
                    ).items()
                }

                fig = self.plot_interaction_network(
                    interactions["player_interactions"],
                    player_positions,
                    game_data.get("player_teams", {}),
                    title="Player Interaction Network",
                )
                plot_json = json.dumps(fig.to_dict())
                html += f"Plotly.newPlot('interaction-network', {plot_json}.data, {plot_json}.layout);\n"

            # Add event timeline
            if events:
                fig = self.plot_event_timeline(events)
                plot_json = json.dumps(fig.to_dict())
                html += f"Plotly.newPlot('event-timeline', {plot_json}.data, {plot_json}.layout);\n"

            html += """
                </script>
            </body>
            </html>
            """

            # Save HTML file
            with open(filename, "w") as f:
                f.write(html)

            return True

        except Exception as e:
            print(f"Error creating analysis report: {e}")
            return False


class SportsAnalyticsGUI:
    def __init__(self, root):
        """
        Initialize the GUI

        Args:
            root (tk.Tk): Root Tkinter window
        """
        self.root = root
        self.root.title("AI-Powered Sports Analytics & Player Tracking")
        self.root.geometry("1200x800")

        # Set theme
        self.theme = CONFIG["default_ui_theme"]
        self.set_theme(self.theme)

        # Create tracker and analyzer
        self.tracker = None
        self.analyzer = GameAnalyzer()
        self.visualizer = Visualizer()

        # Video source
        self.video_source = None
        self.is_processing = False
        self.stop_processing_event = threading.Event()
        self.output_frame_queue = queue.Queue(maxsize=10)
        self.frame_update_interval = 50  # ms

        # Create UI elements
        self.create_menu()
        self.create_main_interface()

        # Start UI update loop
        self.update_ui()

    def set_theme(self, theme):
        """Set the UI theme (light or dark)"""
        if theme == "dark":
            bg_color = "#333333"
            fg_color = "#FFFFFF"
            button_bg = "#555555"
            button_fg = "#FFFFFF"
            frame_bg = "#444444"
            entry_bg = "#666666"
            entry_fg = "#FFFFFF"
        else:  # light
            bg_color = "#F0F0F0"
            fg_color = "#000000"
            button_bg = "#E0E0E0"
            button_fg = "#000000"
            frame_bg = "#F5F5F5"
            entry_bg = "#FFFFFF"
            entry_fg = "#000000"

        self.root.configure(bg=bg_color)
        self.theme_colors = {
            "bg": bg_color,
            "fg": fg_color,
            "button_bg": button_bg,
            "button_fg": button_fg,
            "frame_bg": frame_bg,
            "entry_bg": entry_bg,
            "entry_fg": entry_fg,
        }

        style = ttk.Style()
        style.configure("TButton", background=button_bg, foreground=button_fg)
        style.configure("TLabel", background=bg_color, foreground=fg_color)
        style.configure("TFrame", background=frame_bg)
        style.configure("TEntry", fieldbackground=entry_bg, foreground=entry_fg)
        style.configure("TCheckbutton", background=bg_color, foreground=fg_color)
        style.configure("TRadiobutton", background=bg_color, foreground=fg_color)
        style.configure("TProgressbar", background="blue")

    def stop_processing(self):
        """Stop video processing"""
        if not self.is_processing:
            return

        # Set stop flag
        self.stop_processing_event.set()

        # Wait for thread to finish
        if hasattr(self, "processing_thread") and self.processing_thread.is_alive():
            self.processing_thread.join(timeout=1.0)

        # Release video source
        if self.video_source:
            self.video_source.release()

        # Update UI
        self.is_processing = False
        self.start_button.config(state=tk.NORMAL)
        self.stop_button.config(state=tk.DISABLED)
        self.status_var.set("Processing stopped.")

        # Add current game to analyzer
        if self.tracker:
            game_data = {
                "player_tracks": self.tracker.players,
                "player_speeds": self.tracker.player_speeds,
                "player_distances": self.tracker.player_distances,
                "player_actions": self.tracker.player_actions,
                "player_teams": self.tracker.player_teams,
                "events": self.tracker.events,
                "possession_data": self.tracker.possession_data,
                "zone_data": self.tracker.zone_data,
                "frame_positions": self.tracker.player_positions,
                "total_frames": self.tracker.current_frame,
                "frame_rate": self.tracker.frame_rate,
            }

            self.analyzer.add_game("current_game", game_data)

    def create_menu(self):
        """Create the application menu"""
        # Create menu bar
        menu_bar = tk.Menu(self.root)

        # File menu
        file_menu = tk.Menu(menu_bar, tearoff=0)
        file_menu.add_command(label="Open Video", command=self.open_video)
        file_menu.add_command(
            label="Open Analysis Data", command=self.open_analysis_data
        )
        file_menu.add_separator()
        file_menu.add_command(label="Exit", command=self.root.quit)
        menu_bar.add_cascade(label="File", menu=file_menu)

        # Video Source menu
        source_menu = tk.Menu(menu_bar, tearoff=0)
        source_menu.add_command(
            label="Video File", command=lambda: self.set_video_source("file")
        )
        source_menu.add_command(
            label="Webcam", command=lambda: self.set_video_source("webcam")
        )
        source_menu.add_command(
            label="IP Camera", command=lambda: self.set_video_source("ip_camera")
        )
        menu_bar.add_cascade(label="Video Source", menu=source_menu)

        # Analysis menu
        analysis_menu = tk.Menu(menu_bar, tearoff=0)
        analysis_menu.add_command(label="Track Players", command=self.track_players)
        analysis_menu.add_command(label="Generate Report", command=self.generate_report)
        analysis_menu.add_separator()
        analysis_menu.add_command(
            label="Player Statistics", command=self.show_player_stats
        )
        analysis_menu.add_command(
            label="Team Analysis", command=self.show_team_analysis
        )
        analysis_menu.add_command(
            label="Event Timeline", command=self.show_event_timeline
        )
        menu_bar.add_cascade(label="Analysis", menu=analysis_menu)

        # Visualization menu
        vis_menu = tk.Menu(menu_bar, tearoff=0)
        vis_menu.add_command(label="Heatmaps", command=self.show_heatmaps)
        vis_menu.add_command(label="Trajectories", command=self.show_trajectories)
        vis_menu.add_command(label="Speed Profiles", command=self.show_speed_profiles)
        vis_menu.add_command(
            label="Interaction Network", command=self.show_interaction_network
        )
        menu_bar.add_cascade(label="Visualization", menu=vis_menu)

        # Settings menu
        settings_menu = tk.Menu(menu_bar, tearoff=0)

        # Sport submenu
        sport_menu = tk.Menu(settings_menu, tearoff=0)
        self.selected_sport = tk.StringVar(value="football")
        for sport in CONFIG["sports"].keys():
            sport_menu.add_radiobutton(
                label=sport.capitalize(), variable=self.selected_sport, value=sport
            )
        settings_menu.add_cascade(label="Sport", menu=sport_menu)

        # Theme submenu
        theme_menu = tk.Menu(settings_menu, tearoff=0)
        theme_menu.add_command(
            label="Light Theme", command=lambda: self.set_theme("light")
        )
        theme_menu.add_command(
            label="Dark Theme", command=lambda: self.set_theme("dark")
        )
        settings_menu.add_cascade(label="Theme", menu=theme_menu)

        # Settings options
        settings_menu.add_separator()
        settings_menu.add_command(
            label="Advanced Settings", command=self.show_advanced_settings
        )
        menu_bar.add_cascade(label="Settings", menu=settings_menu)

        # Help menu
        help_menu = tk.Menu(menu_bar, tearoff=0)
        help_menu.add_command(label="Documentation", command=self.show_documentation)
        help_menu.add_command(label="About", command=self.show_about)
        menu_bar.add_cascade(label="Help", menu=help_menu)

        self.root.config(menu=menu_bar)

    def _process_video_optimized(
        self,
        video_path,
        output_path,
        detect_teams=True,
        target_resolution=None,
        progress_callback=None,
    ):
        """
        Process a video file with optimized settings for smoother performance

        Args:
            video_path (str): Path to input video
            output_path (str): Path for output video (or None)
            detect_teams (bool): Whether to detect teams
            target_resolution (tuple): Target resolution for processing (width, height) or None for original
            progress_callback (function): Callback for progress updates

        Returns:
            dict: Analysis results
        """
        # Load video
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Could not open video file: {video_path}")

        # Get video properties
        orig_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        orig_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        # Determine processing resolution
        if target_resolution:
            proc_width, proc_height = target_resolution
            # Keep aspect ratio
            aspect_ratio = orig_width / orig_height
            if proc_width / proc_height > aspect_ratio:
                proc_width = int(proc_height * aspect_ratio)
            else:
                proc_height = int(proc_width / aspect_ratio)
        else:
            proc_width, proc_height = orig_width, orig_height

        # Set up tracker with appropriate settings
        self.tracker.frame_width = orig_width
        self.tracker.frame_height = orig_height
        self.tracker.frame_rate = fps
        self.tracker.set_field_scale((proc_width, proc_height))

        # Set up output video if needed
        out = None
        if output_path:
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            out = cv2.VideoWriter(output_path, fourcc, fps, (orig_width, orig_height))

        # Process in chunks to avoid memory issues
        chunk_size = 5 * int(fps)  # Process 5 seconds at a time
        frame_count = 0
        last_progress = -1
        processing_start_time = time.time()

        # Store frame numbers for each chunk to sync results
        chunk_frames = []

        try:
            while True:
                # Process a chunk of frames
                frames_to_process = []
                frame_indices = []

                # Read frames for this chunk
                for _ in range(chunk_size):
                    ret, frame = cap.read()
                    if not ret:
                        break

                    frame_count += 1
                    frame_indices.append(frame_count)

                    # Resize frame for processing if needed
                    if proc_width != orig_width or proc_height != orig_height:
                        proc_frame = cv2.resize(frame, (proc_width, proc_height))
                    else:
                        proc_frame = frame

                    frames_to_process.append((frame, proc_frame))

                    # Update progress
                    if total_frames:
                        progress = int((frame_count / total_frames) * 100)
                        if progress != last_progress and progress % 5 == 0:
                            if progress_callback:
                                progress_callback(progress)
                            last_progress = progress

                # If no frames read, we're done
                if not frames_to_process:
                    break

                # Store frame indices for this chunk
                chunk_frames.append(frame_indices)

                # Process all frames in this chunk
                for i, (orig_frame, proc_frame) in enumerate(frames_to_process):
                    # Process the frame (tracking, detection, etc.)
                    annotated_frame, results = self.tracker.process_frame(
                        proc_frame, detect_teams
                    )

                    # Resize annotated frame back to original size if needed
                    if annotated_frame is not None and (
                        proc_width != orig_width or proc_height != orig_height
                    ):
                        annotated_frame = cv2.resize(
                            annotated_frame, (orig_width, orig_height)
                        )

                    # Save to output video if requested
                    if out and annotated_frame is not None:
                        out.write(annotated_frame)

        finally:
            # Clean up
            cap.release()
            if out:
                out.release()

        processing_duration = time.time() - processing_start_time

        # Return analysis results
        analysis_results = {
            "player_tracks": self.tracker.players,
            "player_speeds": self.tracker.player_speeds,
            "player_distances": self.tracker.player_distances,
            "player_actions": self.tracker.player_actions,
            "frame_positions": self.tracker.player_positions,
            "player_teams": self.tracker.player_teams,
            "events": self.tracker.events,
            "possession_data": self.tracker.possession_data,
            "zone_data": self.tracker.zone_data,
            "total_frames": frame_count,
            "frame_rate": fps,
            "processing_duration": processing_duration,
        }

        # Save analysis results
        results_path = os.path.join(
            CONFIG["output_dir"], "data", f"analysis_{int(time.time())}.pkl"
        )
        with open(results_path, "wb") as f:
            pickle.dump(analysis_results, f)

        return analysis_results

    def create_main_interface(self):
        """Create the main application interface"""
        # Main frame
        main_frame = ttk.Frame(self.root, padding=10)
        main_frame.pack(fill=tk.BOTH, expand=True)

        # Create left panel (video display and controls)
        left_panel = ttk.Frame(main_frame)
        left_panel.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        # Add a fixed size to the video frame and make it non-resizable
        self.video_frame = ttk.Frame(
            left_panel, borderwidth=2, relief="groove", width=640, height=480
        )
        self.video_frame.pack(fill=tk.BOTH, expand=True, pady=10)
        self.video_frame.pack_propagate(False)  # This prevents the frame from resizing

        # Create canvas for video display with fixed size
        self.canvas = tk.Canvas(self.video_frame, bg="black", width=640, height=480)
        self.canvas.pack(fill=tk.BOTH, expand=True)

        # Video controls
        controls_frame = ttk.Frame(left_panel)
        controls_frame.pack(fill=tk.X, pady=10)

        # Source selection
        source_frame = ttk.LabelFrame(controls_frame, text="Video Source")
        source_frame.pack(fill=tk.X, pady=5)

        source_buttons_frame = ttk.Frame(source_frame)
        source_buttons_frame.pack(fill=tk.X, padx=5, pady=5)

        ttk.Button(
            source_buttons_frame,
            text="Video File",
            command=lambda: self.set_video_source("file"),
        ).pack(side=tk.LEFT, padx=5)
        ttk.Button(
            source_buttons_frame,
            text="Webcam",
            command=lambda: self.set_video_source("webcam"),
        ).pack(side=tk.LEFT, padx=5)
        ttk.Button(
            source_buttons_frame,
            text="IP Camera",
            command=lambda: self.set_video_source("ip_camera"),
        ).pack(side=tk.LEFT, padx=5)

        # Processing controls
        process_frame = ttk.LabelFrame(controls_frame, text="Processing")
        process_frame.pack(fill=tk.X, pady=5)

        process_buttons_frame = ttk.Frame(process_frame)
        process_buttons_frame.pack(fill=tk.X, padx=5, pady=5)

        self.start_button = ttk.Button(
            process_buttons_frame,
            text="Start Processing",
            command=self.start_processing,
        )
        self.start_button.pack(side=tk.LEFT, padx=5)

        self.stop_button = ttk.Button(
            process_buttons_frame,
            text="Stop Processing",
            command=self.stop_processing,
            state=tk.DISABLED,
        )
        self.stop_button.pack(side=tk.LEFT, padx=5)

        ttk.Button(
            process_buttons_frame, text="Save Results", command=self.save_results
        ).pack(side=tk.LEFT, padx=5)

        # Progress bar
        self.progress_var = tk.DoubleVar()
        self.progress_bar = ttk.Progressbar(
            controls_frame, variable=self.progress_var, maximum=100, mode="determinate"
        )
        self.progress_bar.pack(fill=tk.X, pady=5)

        # Status label
        self.status_var = tk.StringVar(value="Ready")
        status_label = ttk.Label(
            controls_frame, textvariable=self.status_var, anchor=tk.W
        )
        status_label.pack(fill=tk.X, pady=5)

        # Create right panel (analysis and visualization)
        right_panel = ttk.Frame(main_frame, width=400)
        right_panel.pack(side=tk.RIGHT, fill=tk.BOTH, padx=10)
        right_panel.pack_propagate(False)  # Prevent shrinking

        # Analysis section
        analysis_frame = ttk.LabelFrame(right_panel, text="Analysis")
        analysis_frame.pack(fill=tk.X, pady=5)

        # Sport selection
        sport_frame = ttk.Frame(analysis_frame)
        sport_frame.pack(fill=tk.X, padx=5, pady=5)

        ttk.Label(sport_frame, text="Sport:").pack(side=tk.LEFT, padx=5)
        sport_combo = ttk.Combobox(
            sport_frame,
            textvariable=self.selected_sport,
            state="readonly",
            values=list(CONFIG["sports"].keys()),
        )
        sport_combo.pack(side=tk.LEFT, padx=5, fill=tk.X, expand=True)

        # Detection settings
        detect_frame = ttk.Frame(analysis_frame)
        detect_frame.pack(fill=tk.X, padx=5, pady=5)

        self.detect_teams_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(
            detect_frame, text="Detect Teams", variable=self.detect_teams_var
        ).pack(side=tk.LEFT, padx=5)

        self.detect_actions_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(
            detect_frame, text="Detect Actions", variable=self.detect_actions_var
        ).pack(side=tk.LEFT, padx=5)

        # Analysis buttons
        analysis_buttons_frame = ttk.Frame(analysis_frame)
        analysis_buttons_frame.pack(fill=tk.X, padx=5, pady=5)

        ttk.Button(
            analysis_buttons_frame, text="Player Stats", command=self.show_player_stats
        ).pack(side=tk.LEFT, padx=5)
        ttk.Button(
            analysis_buttons_frame,
            text="Team Analysis",
            command=self.show_team_analysis,
        ).pack(side=tk.LEFT, padx=5)
        ttk.Button(
            analysis_buttons_frame,
            text="Event Timeline",
            command=self.show_event_timeline,
        ).pack(side=tk.LEFT, padx=5)

        # Visualization section
        vis_frame = ttk.LabelFrame(right_panel, text="Visualization")
        vis_frame.pack(fill=tk.X, pady=5)

        # Visualization buttons
        vis_buttons_frame1 = ttk.Frame(vis_frame)
        vis_buttons_frame1.pack(fill=tk.X, padx=5, pady=5)

        ttk.Button(
            vis_buttons_frame1, text="Heatmaps", command=self.show_heatmaps
        ).pack(side=tk.LEFT, padx=5)
        ttk.Button(
            vis_buttons_frame1, text="Trajectories", command=self.show_trajectories
        ).pack(side=tk.LEFT, padx=5)

        vis_buttons_frame2 = ttk.Frame(vis_frame)
        vis_buttons_frame2.pack(fill=tk.X, padx=5, pady=5)

        ttk.Button(
            vis_buttons_frame2, text="Speed Profiles", command=self.show_speed_profiles
        ).pack(side=tk.LEFT, padx=5)
        ttk.Button(
            vis_buttons_frame2, text="Network", command=self.show_interaction_network
        ).pack(side=tk.LEFT, padx=5)

        # Report section
        report_frame = ttk.LabelFrame(right_panel, text="Reports")
        report_frame.pack(fill=tk.X, pady=5)

        report_buttons_frame = ttk.Frame(report_frame)
        report_buttons_frame.pack(fill=tk.X, padx=5, pady=5)

        ttk.Button(
            report_buttons_frame, text="Generate Report", command=self.generate_report
        ).pack(side=tk.LEFT, padx=5)
        ttk.Button(
            report_buttons_frame, text="Export Data", command=self.export_data
        ).pack(side=tk.LEFT, padx=5)

        # Results section - shows tracked entities and stats
        results_frame = ttk.LabelFrame(right_panel, text="Results")
        results_frame.pack(fill=tk.BOTH, expand=True, pady=5)

        # Create a canvas with scrollbar for results
        results_canvas = tk.Canvas(results_frame)
        results_scrollbar = ttk.Scrollbar(
            results_frame, orient="vertical", command=results_canvas.yview
        )
        results_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        results_canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        results_canvas.configure(yscrollcommand=results_scrollbar.set)

        # Frame inside canvas for results content
        self.results_content = ttk.Frame(results_canvas)
        results_canvas.create_window((0, 0), window=self.results_content, anchor="nw")

        # Configure the canvas to resize with the frame
        def configure_results_canvas(event):
            results_canvas.configure(
                scrollregion=results_canvas.bbox("all"), width=event.width
            )

        self.results_content.bind("<Configure>", configure_results_canvas)

        # Initial results display
        ttk.Label(self.results_content, text="No results available yet.").pack(pady=10)

    def update_ui(self):
        """Update UI elements periodically"""
        # Update video frame if queue has items
        if not self.output_frame_queue.empty():
            try:
                frame = self.output_frame_queue.get_nowait()
                self.display_frame(frame)
            except queue.Empty:
                pass

        # Update results display if needed
        if hasattr(self, "tracker") and self.tracker and self.is_processing:
            self.update_results_display()

        # Schedule next update
        self.root.after(self.frame_update_interval, self.update_ui)

    def display_frame(self, frame):
        """Display a frame on the canvas"""
        if frame is None:
            return

        # Get current canvas size
        canvas_width = self.canvas.winfo_width()
        canvas_height = self.canvas.winfo_height()

        # Use minimum fixed size if canvas isn't properly sized yet
        if canvas_width < 10 or canvas_height < 10:
            canvas_width = 640
            canvas_height = 480

        # Calculate original aspect ratio
        frame_height, frame_width = frame.shape[:2]
        aspect_ratio = frame_width / frame_height

        # Determine target dimensions without changing canvas size
        if canvas_width / canvas_height > aspect_ratio:
            # Canvas is wider than frame
            target_height = canvas_height
            target_width = int(target_height * aspect_ratio)
        else:
            # Canvas is taller than frame
            target_width = canvas_width
            target_height = int(target_width / aspect_ratio)

        # Resize frame
        frame = cv2.resize(frame, (target_width, target_height))

        # Convert frame to RGB for tkinter
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Convert to ImageTk format
        img = Image.fromarray(frame_rgb)
        img_tk = ImageTk.PhotoImage(image=img)

        # Clear previous content and update canvas without changing its size
        self.canvas.delete("all")
        self.canvas.create_image(
            canvas_width // 2, canvas_height // 2, anchor=tk.CENTER, image=img_tk
        )
        self.canvas.image = img_tk  # Keep a reference

    def update_results_display(self):
        """Update the results display with current tracking data"""
        # Clear current results
        for widget in self.results_content.winfo_children():
            widget.destroy()

        if not hasattr(self, "tracker") or not self.tracker:
            ttk.Label(self.results_content, text="No tracking data available.").pack(
                pady=10
            )
            return

        # Show tracked players
        ttk.Label(
            self.results_content, text="Tracked Players", font=("-weight bold")
        ).pack(anchor=tk.W, pady=(10, 5))

        if not self.tracker.players:
            ttk.Label(self.results_content, text="No players detected yet.").pack(
                anchor=tk.W, padx=10
            )
        else:
            # Create scrollable player list
            player_frame = ttk.Frame(self.results_content)
            player_frame.pack(fill=tk.X, expand=True, padx=5)

            row = 0
            columns = (
                "ID",
                "Team",
                "Distance (m)",
                "Avg Speed (m/s)",
                "Max Speed (m/s)",
            )
            for col_idx, col in enumerate(columns):
                ttk.Label(player_frame, text=col, font=("-weight bold")).grid(
                    row=row, column=col_idx, sticky=tk.W, padx=5, pady=2
                )

            row += 1
            for player_id, positions in self.tracker.players.items():
                if len(positions) < 2:
                    continue

                # Get player stats
                distance = self.tracker.player_distances.get(player_id, 0)
                speeds = self.tracker.player_speeds.get(player_id, [])
                avg_speed = sum(speeds) / len(speeds) if speeds else 0
                max_speed = max(speeds) if speeds else 0
                team = self.tracker.player_teams.get(player_id, -1)
                team_str = f"{team+1}" if team != -1 else "-"

                # Add player row
                ttk.Label(player_frame, text=str(player_id)).grid(
                    row=row, column=0, sticky=tk.W, padx=5, pady=2
                )
                ttk.Label(player_frame, text=team_str).grid(
                    row=row, column=1, sticky=tk.W, padx=5, pady=2
                )
                ttk.Label(player_frame, text=f"{distance:.2f}").grid(
                    row=row, column=2, sticky=tk.W, padx=5, pady=2
                )
                ttk.Label(player_frame, text=f"{avg_speed:.2f}").grid(
                    row=row, column=3, sticky=tk.W, padx=5, pady=2
                )
                ttk.Label(player_frame, text=f"{max_speed:.2f}").grid(
                    row=row, column=4, sticky=tk.W, padx=5, pady=2
                )

                row += 1

        # Show recent events
        ttk.Label(
            self.results_content, text="Recent Events", font=("-weight bold")
        ).pack(anchor=tk.W, pady=(15, 5))

        if not self.tracker.events:
            ttk.Label(self.results_content, text="No events detected yet.").pack(
                anchor=tk.W, padx=10
            )
        else:
            # Show last 5 events
            recent_events = sorted(
                self.tracker.events, key=lambda e: e.get("frame", 0), reverse=True
            )[:5]

            event_frame = ttk.Frame(self.results_content)
            event_frame.pack(fill=tk.X, expand=True, padx=5)

            row = 0
            columns = ("Time", "Action", "Team", "Player")
            for col_idx, col in enumerate(columns):
                ttk.Label(event_frame, text=col, font=("-weight bold")).grid(
                    row=row, column=col_idx, sticky=tk.W, padx=5, pady=2
                )

            row += 1
            for event in recent_events:
                # Calculate time
                frame = event.get("frame", 0)
                time_seconds = (
                    frame / self.tracker.frame_rate if self.tracker.frame_rate else 0
                )
                minutes = int(time_seconds // 60)
                seconds = int(time_seconds % 60)
                time_str = f"{minutes:02d}:{seconds:02d}"

                # Get event info
                action = event.get("action", "Unknown")
                team = event.get("team", -1)
                team_str = f"{team+1}" if team != -1 else "-"
                player = event.get("player", None)
                player_str = str(player) if player is not None else "-"

                # Add event row
                ttk.Label(event_frame, text=time_str).grid(
                    row=row, column=0, sticky=tk.W, padx=5, pady=2
                )
                ttk.Label(event_frame, text=action).grid(
                    row=row, column=1, sticky=tk.W, padx=5, pady=2
                )
                ttk.Label(event_frame, text=team_str).grid(
                    row=row, column=2, sticky=tk.W, padx=5, pady=2
                )
                ttk.Label(event_frame, text=player_str).grid(
                    row=row, column=3, sticky=tk.W, padx=5, pady=2
                )

                row += 1

        # Show possession stats if available
        if self.tracker.possession_data:
            ttk.Label(
                self.results_content, text="Possession", font=("-weight bold")
            ).pack(anchor=tk.W, pady=(15, 5))

            # Calculate possession percentages
            team_frames = defaultdict(int)
            for frame, data in self.tracker.possession_data.items():
                team = data.get("team", -1)
                if team != -1:
                    team_frames[team] += 1

            total_frames = sum(team_frames.values())

            if total_frames > 0:
                poss_frame = ttk.Frame(self.results_content)
                poss_frame.pack(fill=tk.X, expand=True, padx=5)

                ttk.Label(poss_frame, text="Team", font=("-weight bold")).grid(
                    row=0, column=0, sticky=tk.W, padx=5, pady=2
                )
                ttk.Label(poss_frame, text="Possession", font=("-weight bold")).grid(
                    row=0, column=1, sticky=tk.W, padx=5, pady=2
                )

                for team, frames in team_frames.items():
                    percentage = (frames / total_frames) * 100
                    ttk.Label(poss_frame, text=f"Team {team+1}").grid(
                        row=team + 1, column=0, sticky=tk.W, padx=5, pady=2
                    )
                    ttk.Label(poss_frame, text=f"{percentage:.1f}%").grid(
                        row=team + 1, column=1, sticky=tk.W, padx=5, pady=2
                    )

    def set_video_source(self, source_type):
        """Set the video source type"""
        if self.is_processing:
            messagebox.showerror(
                "Error", "Please stop processing before changing video source."
            )
            return

        if source_type == "file":
            self.open_video()
        elif source_type == "webcam":
            self.setup_webcam()
        elif source_type == "ip_camera":
            self.setup_ip_camera()

    def open_video(self):
        """Open a video file"""
        file_path = filedialog.askopenfilename(
            title="Select Video File",
            filetypes=[
                ("Video files", "*.mp4 *.avi *.mov *.mkv"),
                ("All files", "*.*"),
            ],
        )

        if not file_path:
            return

        # Create video source
        try:
            self.video_source = VideoSource(source_type="file", source_path=file_path)
            if self.video_source.connect():
                video_info = self.video_source.get_video_info()
                self.status_var.set(
                    f"Loaded video: {os.path.basename(file_path)}, "
                    + f"{video_info['width']}x{video_info['height']}, "
                    + f"{video_info['fps']} fps, {video_info['duration']} seconds"
                )

                # Show first frame
                ret, frame = self.video_source.read_frame()
                if ret:
                    self.display_frame(frame)

                # Reset video source
                self.video_source.release()
                self.video_source = VideoSource(
                    source_type="file", source_path=file_path
                )
            else:
                messagebox.showerror("Error", "Could not open video file.")
        except Exception as e:
            messagebox.showerror("Error", f"Error loading video: {str(e)}")

    def setup_webcam(self):
        """Setup webcam as video source"""
        # Create dialog to select webcam device
        dialog = tk.Toplevel(self.root)
        dialog.title("Webcam Setup")
        dialog.geometry("400x200")
        dialog.transient(self.root)
        dialog.grab_set()

        ttk.Label(dialog, text="Select Webcam:").pack(pady=(20, 10))

        # Device selection
        device_var = tk.StringVar(value="0")
        device_frame = ttk.Frame(dialog)
        device_frame.pack(pady=10)

        ttk.Label(device_frame, text="Device:").grid(row=0, column=0, padx=5)
        device_entry = ttk.Entry(device_frame, textvariable=device_var, width=10)
        device_entry.grid(row=0, column=1, padx=5)

        # Resolution selection
        resolution_var = tk.StringVar(value="1280x720")
        resolutions = ["640x480", "1280x720", "1920x1080"]

        ttk.Label(device_frame, text="Resolution:").grid(
            row=1, column=0, padx=5, pady=10
        )
        resolution_combo = ttk.Combobox(
            device_frame,
            textvariable=resolution_var,
            values=resolutions,
            state="readonly",
        )
        resolution_combo.grid(row=1, column=1, padx=5, pady=10)

        # Buttons
        button_frame = ttk.Frame(dialog)
        button_frame.pack(pady=20)

        def on_connect():
            try:
                device_idx = int(device_var.get())
                width, height = map(int, resolution_var.get().split("x"))

                self.video_source = VideoSource(
                    source_type="webcam",
                    source_path=device_idx,
                    resolution=(width, height),
                )

                if self.video_source.connect():
                    video_info = self.video_source.get_video_info()
                    self.status_var.set(
                        f"Connected to webcam: device {device_idx}, "
                        + f"{video_info['width']}x{video_info['height']}, "
                        + f"{video_info['fps']} fps"
                    )

                    # Show current frame
                    ret, frame = self.video_source.read_frame()
                    if ret:
                        self.display_frame(frame)

                    # Reset video source for later use
                    self.video_source.release()
                    self.video_source = VideoSource(
                        source_type="webcam",
                        source_path=device_idx,
                        resolution=(width, height),
                    )

                    dialog.destroy()
                else:
                    messagebox.showerror(
                        "Error", f"Could not connect to webcam device {device_idx}"
                    )
            except Exception as e:
                messagebox.showerror("Error", f"Error connecting to webcam: {str(e)}")

        ttk.Button(button_frame, text="Connect", command=on_connect).pack(
            side=tk.LEFT, padx=10
        )
        ttk.Button(button_frame, text="Cancel", command=dialog.destroy).pack(
            side=tk.LEFT, padx=10
        )

    def setup_ip_camera(self):
        """Setup IP camera as video source"""
        # Create dialog to enter IP camera details
        dialog = tk.Toplevel(self.root)
        dialog.title("IP Camera Setup")
        dialog.geometry("500x250")
        dialog.transient(self.root)
        dialog.grab_set()

        ttk.Label(dialog, text="Enter IP Camera Details:").pack(pady=(20, 10))

        # Form for camera details
        form_frame = ttk.Frame(dialog)
        form_frame.pack(pady=10, fill=tk.X, padx=20)

        # URL
        url_var = tk.StringVar(
            value=CONFIG["camera_settings"]["ip_camera"]["default_url"]
        )
        ttk.Label(form_frame, text="URL:").grid(row=0, column=0, sticky=tk.W, pady=5)
        url_entry = ttk.Entry(form_frame, textvariable=url_var, width=40)
        url_entry.grid(row=0, column=1, sticky=tk.W + tk.E, pady=5)

        # Username (optional)
        username_var = tk.StringVar(
            value=CONFIG["camera_settings"]["ip_camera"]["username"]
        )
        ttk.Label(form_frame, text="Username (optional):").grid(
            row=1, column=0, sticky=tk.W, pady=5
        )
        username_entry = ttk.Entry(form_frame, textvariable=username_var, width=20)
        username_entry.grid(row=1, column=1, sticky=tk.W, pady=5)

        # Password (optional)
        password_var = tk.StringVar(
            value=CONFIG["camera_settings"]["ip_camera"]["password"]
        )
        ttk.Label(form_frame, text="Password (optional):").grid(
            row=2, column=0, sticky=tk.W, pady=5
        )
        password_entry = ttk.Entry(
            form_frame, textvariable=password_var, width=20, show="*"
        )
        password_entry.grid(row=2, column=1, sticky=tk.W, pady=5)

        # Buttons
        button_frame = ttk.Frame(dialog)
        button_frame.pack(pady=20)

        def on_connect():
            try:
                url = url_var.get()
                username = username_var.get()
                password = password_var.get()

                if not url:
                    messagebox.showerror("Error", "URL is required.")
                    return

                self.video_source = VideoSource(
                    source_type="ip_camera",
                    ip_address=url,
                    username=username,
                    password=password,
                )

                if self.video_source.connect():
                    video_info = self.video_source.get_video_info()
                    self.status_var.set(
                        f"Connected to IP camera: {url}, "
                        + f"{video_info['width']}x{video_info['height']}, "
                        + f"{video_info['fps']} fps"
                    )

                    # Show current frame
                    ret, frame = self.video_source.read_frame()
                    if ret:
                        self.display_frame(frame)

                    # Reset video source for later use
                    self.video_source.release()
                    self.video_source = VideoSource(
                        source_type="ip_camera",
                        ip_address=url,
                        username=username,
                        password=password,
                    )

                    dialog.destroy()
                else:
                    messagebox.showerror(
                        "Error", f"Could not connect to IP camera at {url}"
                    )
            except Exception as e:
                messagebox.showerror(
                    "Error", f"Error connecting to IP camera: {str(e)}"
                )

        ttk.Button(button_frame, text="Connect", command=on_connect).pack(
            side=tk.LEFT, padx=10
        )
        ttk.Button(button_frame, text="Cancel", command=dialog.destroy).pack(
            side=tk.LEFT, padx=10
        )

    def open_analysis_data(self):
        """Open previously saved analysis data"""
        file_path = filedialog.askopenfilename(
            title="Open Analysis Data",
            filetypes=[("Pickle files", "*.pkl"), ("All files", "*.*")],
        )

        if not file_path:
            return

        try:
            with open(file_path, "rb") as f:
                data = pickle.load(f)

            # Create new tracker and set its data
            sport = self.selected_sport.get()
            self.tracker = PlayerTracker(sport=sport)

            # Copy all attributes from loaded data
            for key, value in data.items():
                setattr(self.tracker, key, value)

            # Update UI
            self.status_var.set(
                f"Loaded analysis data from {os.path.basename(file_path)}"
            )
            self.update_results_display()

            # Enable analysis buttons
            self.analyzer.add_game("current_game", data)

            messagebox.showinfo("Success", "Analysis data loaded successfully.")
        except Exception as e:
            messagebox.showerror("Error", f"Error loading analysis data: {str(e)}")

    def start_processing(self):
        """Start video processing"""
        if self.is_processing:
            return

        if not self.video_source:
            messagebox.showerror("Error", "No video source selected.")
            return

        # Create new tracker
        sport = self.selected_sport.get()
        self.tracker = PlayerTracker(sport=sport)

        # Connect to video source
        if not self.video_source.connect():
            messagebox.showerror("Error", "Could not connect to video source.")
            return

        # Get video info for tracker
        video_info = self.video_source.get_video_info()
        self.tracker.frame_width = video_info["width"]
        self.tracker.frame_height = video_info["height"]
        self.tracker.frame_rate = video_info["fps"]

        # Set field scale
        self.tracker.set_field_scale(
            (self.tracker.frame_width, self.tracker.frame_height)
        )

        # Clear existing frames in queue
        while not self.output_frame_queue.empty():
            try:
                self.output_frame_queue.get_nowait()
            except queue.Empty:
                break

        # Reset processing flags
        self.is_processing = True
        self.stop_processing_event.clear()

        # Update UI
        self.start_button.config(state=tk.DISABLED)
        self.stop_button.config(state=tk.NORMAL)
        self.status_var.set("Processing video...")
        self.progress_var.set(0)

        # Start processing thread with memory limits for queue
        self.processing_thread = threading.Thread(
            target=self._process_stream_with_throttling,
            args=(
                self.video_source,
                self.stop_processing_event,
                self.detect_teams_var.get(),
            ),
        )
        self.processing_thread.daemon = True
        self.processing_thread.start()

    def _process_stream_with_throttling(
        self, video_source, stop_event, detect_teams=True
    ):
        """Process video stream with throttling to maintain performance"""
        frame_count = 0
        skip_frames = 0  # For throttling if needed

        try:
            while not stop_event.is_set():
                # Read frame
                ret, frame = video_source.read_frame()
                if not ret:
                    if video_source.source_type == "ip_camera":
                        print(
                            "Lost connection to IP camera. Attempting to reconnect..."
                        )
                        if video_source.connect():
                            continue
                    break

                frame_count += 1

                # Skip frames if processing is falling behind
                if (
                    self.output_frame_queue.qsize() > 3
                ):  # If queue has several frames waiting
                    skip_frames = min(skip_frames + 1, 3)  # Increase skip rate up to 3
                else:
                    skip_frames = max(skip_frames - 1, 0)  # Decrease skip rate

                if skip_frames > 0 and frame_count % (skip_frames + 1) != 0:
                    continue  # Skip this frame

                # Process frame
                start_time = time.time()

                # Resize frame if it's very large (for performance)
                h, w = frame.shape[:2]
                if w > 1280 or h > 720:
                    scale = min(1280 / w, 720 / h)
                    new_w, new_h = int(w * scale), int(h * scale)
                    processed_frame = cv2.resize(frame, (new_w, new_h))
                else:
                    processed_frame = frame

                # Process the frame
                annotated_frame, results = self.tracker.process_frame(
                    processed_frame, detect_teams
                )

                if annotated_frame is None:
                    continue

                # Add analysis overlay
                annotated_frame = self.tracker.add_analysis_overlay(annotated_frame)

                # Limit queue size to prevent memory issues
                if self.output_frame_queue.qsize() < 5:  # Limit queue to 5 frames max
                    self.output_frame_queue.put(annotated_frame)

                # Throttle processing rate
                processing_time = time.time() - start_time
                frame_time = 1.0 / CONFIG["max_fps_processing"]
                if processing_time < frame_time:
                    time.sleep(frame_time - processing_time)

        except Exception as e:
            print(f"Error in video processing: {e}")

        finally:
            # Clean up
            if video_source:
                video_source.release()

    def track_players(self):
        """Process a video file for player tracking"""
        if self.is_processing:
            messagebox.showerror("Error", "Already processing video.")
            return

        file_path = filedialog.askopenfilename(
            title="Select Video File",
            filetypes=[
                ("Video files", "*.mp4 *.avi *.mov *.mkv"),
                ("All files", "*.*"),
            ],
        )

        if not file_path:
            return

        # Ask for output options
        output_dir = os.path.join(CONFIG["output_dir"], "videos")
        os.makedirs(output_dir, exist_ok=True)

        output_filename = (
            os.path.splitext(os.path.basename(file_path))[0] + "_tracked.mp4"
        )
        output_path = os.path.join(output_dir, output_filename)

        # Create dialog for processing options
        dialog = tk.Toplevel(self.root)
        dialog.title("Processing Options")
        dialog.geometry("500x350")  # Increase height for more options
        dialog.transient(self.root)
        dialog.grab_set()

        # Output file option
        output_var = tk.BooleanVar(value=True)
        output_frame = ttk.Frame(dialog)
        output_frame.pack(fill=tk.X, padx=20, pady=10)

        ttk.Checkbutton(
            output_frame, text="Save tracked video", variable=output_var
        ).pack(anchor=tk.W)

        output_path_var = tk.StringVar(value=output_path)
        path_frame = ttk.Frame(output_frame)
        path_frame.pack(fill=tk.X, pady=5)

        ttk.Label(path_frame, text="Output:").pack(side=tk.LEFT, padx=5)
        ttk.Entry(path_frame, textvariable=output_path_var, width=50).pack(
            side=tk.LEFT, padx=5, fill=tk.X, expand=True
        )

        # Processing options
        options_frame = ttk.LabelFrame(dialog, text="Options")
        options_frame.pack(fill=tk.X, padx=20, pady=10)

        detect_teams_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(
            options_frame, text="Detect Teams", variable=detect_teams_var
        ).pack(anchor=tk.W, padx=10, pady=5)

        # Add processing mode options
        processing_mode_frame = ttk.Frame(options_frame)
        processing_mode_frame.pack(fill=tk.X, padx=10, pady=5)

        ttk.Label(processing_mode_frame, text="Processing Mode:").pack(anchor=tk.W)

        mode_var = tk.StringVar(value="optimized")
        ttk.Radiobutton(
            processing_mode_frame,
            text="Optimized (Recommended for longer videos)",
            variable=mode_var,
            value="optimized",
        ).pack(anchor=tk.W, padx=20, pady=2)
        ttk.Radiobutton(
            processing_mode_frame,
            text="Full Quality (May be slower)",
            variable=mode_var,
            value="full",
        ).pack(anchor=tk.W, padx=20, pady=2)

        # Processing resolution option for optimized mode
        resolution_frame = ttk.Frame(options_frame)
        resolution_frame.pack(fill=tk.X, padx=10, pady=5)

        ttk.Label(resolution_frame, text="Processing Resolution:").pack(anchor=tk.W)

        resolution_var = tk.StringVar(value="720p")
        resolutions = ["360p", "480p", "720p", "1080p", "Original"]
        resolution_combo = ttk.Combobox(
            resolution_frame,
            textvariable=resolution_var,
            values=resolutions,
            state="readonly",
            width=10,
        )
        resolution_combo.pack(anchor=tk.W, padx=20, pady=2)

        # Sport selection
        sport_frame = ttk.Frame(options_frame)
        sport_frame.pack(fill=tk.X, padx=10, pady=5)

        ttk.Label(sport_frame, text="Sport:").pack(side=tk.LEFT, padx=5)
        sport_combo = ttk.Combobox(
            sport_frame,
            textvariable=self.selected_sport,
            state="readonly",
            values=list(CONFIG["sports"].keys()),
        )
        sport_combo.pack(side=tk.LEFT, padx=5, fill=tk.X, expand=True)

        # Buttons
        button_frame = ttk.Frame(dialog)
        button_frame.pack(pady=20)

        def on_start():
            try:
                # Create new tracker
                sport = self.selected_sport.get()
                self.tracker = PlayerTracker(sport=sport)

                # Get output path
                final_output_path = output_path_var.get() if output_var.get() else None
                detect_teams = detect_teams_var.get()
                processing_mode = mode_var.get()
                resolution_setting = resolution_var.get()

                dialog.destroy()

                # Show progress dialog
                progress_dialog = tk.Toplevel(self.root)
                progress_dialog.title("Processing Video")
                progress_dialog.geometry("400x150")
                progress_dialog.transient(self.root)
                progress_dialog.grab_set()

                ttk.Label(
                    progress_dialog,
                    text=f"Processing video: {os.path.basename(file_path)}",
                ).pack(pady=(20, 10))

                progress_var = tk.DoubleVar()
                progress_bar = ttk.Progressbar(
                    progress_dialog,
                    variable=progress_var,
                    maximum=100,
                    mode="determinate",
                )
                progress_bar.pack(fill=tk.X, padx=20, pady=10)

                status_var = tk.StringVar(value="Starting...")
                status_label = ttk.Label(progress_dialog, textvariable=status_var)
                status_label.pack(pady=5)

                # Start processing in a thread
                def update_progress(percent):
                    progress_var.set(percent)
                    status_var.set(f"Processing: {percent}% complete")
                    progress_dialog.update_idletasks()

                def process_thread():
                    try:
                        # Determine processing options based on mode
                        if processing_mode == "optimized":
                            # Determine target resolution
                            target_resolution = None
                            if resolution_setting == "360p":
                                target_resolution = (640, 360)
                            elif resolution_setting == "480p":
                                target_resolution = (854, 480)
                            elif resolution_setting == "720p":
                                target_resolution = (1280, 720)
                            elif resolution_setting == "1080p":
                                target_resolution = (1920, 1080)
                            # Original resolution uses None

                            # Process the video with optimized settings
                            results = self._process_video_optimized(
                                file_path,
                                final_output_path,
                                detect_teams=detect_teams,
                                target_resolution=target_resolution,
                                progress_callback=update_progress,
                            )
                        else:
                            # Process with full quality
                            results = self.tracker.process_video(
                                file_path,
                                final_output_path,
                                save_frames=False,
                                detect_teams=detect_teams,
                                progress_callback=update_progress,
                            )

                        # Add to analyzer
                        if results:
                            self.analyzer.add_game("current_game", results)

                        # Close dialog and update UI
                        progress_dialog.destroy()
                        self.status_var.set(
                            f"Processed video: {os.path.basename(file_path)}"
                        )
                        self.update_results_display()

                        messagebox.showinfo("Success", "Video processing complete!")
                    except Exception as e:
                        progress_dialog.destroy()
                        messagebox.showerror(
                            "Error", f"Error processing video: {str(e)}"
                        )

                threading.Thread(target=process_thread, daemon=True).start()

            except Exception as e:
                dialog.destroy()
                messagebox.showerror("Error", f"Error starting processing: {str(e)}")

        ttk.Button(button_frame, text="Start Processing", command=on_start).pack(
            side=tk.LEFT, padx=10
        )
        ttk.Button(button_frame, text="Cancel", command=dialog.destroy).pack(
            side=tk.LEFT, padx=10
        )

    def save_results(self):
        """Save processing results"""
        if not hasattr(self, "tracker") or not self.tracker:
            messagebox.showerror("Error", "No tracking data available to save.")
            return

        # Ask for save location
        file_path = filedialog.asksaveasfilename(
            title="Save Analysis Data",
            defaultextension=".pkl",
            filetypes=[("Pickle files", "*.pkl"), ("All files", "*.*")],
            initialdir=os.path.join(CONFIG["output_dir"], "data"),
        )

        if not file_path:
            return

        try:
            # Create data dictionary
            # Create data dictionary
            data = {
                "player_tracks": self.tracker.players,
                "player_speeds": self.tracker.player_speeds,
                "player_distances": self.tracker.player_distances,
                "player_actions": self.tracker.player_actions,
                "player_teams": self.tracker.player_teams,
                "events": self.tracker.events,
                "possession_data": self.tracker.possession_data,
                "zone_data": self.tracker.zone_data,
                "frame_positions": self.tracker.player_positions,
                "total_frames": self.tracker.current_frame,
                "frame_rate": self.tracker.frame_rate,
            }

            # Save data
            with open(file_path, "wb") as f:
                pickle.dump(data, f)

            self.status_var.set(f"Saved analysis data to {os.path.basename(file_path)}")
            messagebox.showinfo("Success", "Analysis data saved successfully!")
        except Exception as e:
            messagebox.showerror("Error", f"Error saving data: {str(e)}")

    def export_data(self):
        """Export data to CSV or other formats"""
        if not hasattr(self, "tracker") or not self.tracker:
            messagebox.showerror("Error", "No tracking data available to export.")
            return

        # Create dialog for export options
        dialog = tk.Toplevel(self.root)
        dialog.title("Export Data")
        dialog.geometry("400x300")
        dialog.transient(self.root)
        dialog.grab_set()

        ttk.Label(dialog, text="Select export format:").pack(pady=(20, 10))

        # Format selection
        format_var = tk.StringVar(value="csv")
        format_frame = ttk.Frame(dialog)
        format_frame.pack(pady=10)

        ttk.Radiobutton(
            format_frame, text="CSV (Player tracking)", variable=format_var, value="csv"
        ).pack(anchor=tk.W, pady=5)
        ttk.Radiobutton(
            format_frame, text="JSON (All data)", variable=format_var, value="json"
        ).pack(anchor=tk.W, pady=5)
        ttk.Radiobutton(
            format_frame,
            text="Excel (Stats summary)",
            variable=format_var,
            value="excel",
        ).pack(anchor=tk.W, pady=5)

        # Buttons
        button_frame = ttk.Frame(dialog)
        button_frame.pack(pady=20)

        def on_export():
            export_format = format_var.get()
            dialog.destroy()

            # Ask for save location
            default_extension = f".{export_format}"
            if export_format == "excel":
                default_extension = ".xlsx"

            file_path = filedialog.asksaveasfilename(
                title=f"Export as {export_format.upper()}",
                defaultextension=default_extension,
                filetypes=[
                    (f"{export_format.upper()} files", f"*{default_extension}"),
                    ("All files", "*.*"),
                ],
                initialdir=os.path.join(CONFIG["output_dir"], "data"),
            )

            if not file_path:
                return

            try:
                if export_format == "csv":
                    # Export tracking data to CSV
                    self.tracker.save_tracking_data(file_path)
                elif export_format == "json":
                    # Export all data to JSON
                    data = {
                        "player_tracks": {
                            str(k): v for k, v in self.tracker.players.items()
                        },
                        "player_speeds": {
                            str(k): v for k, v in self.tracker.player_speeds.items()
                        },
                        "player_distances": {
                            str(k): v for k, v in self.tracker.player_distances.items()
                        },
                        "player_teams": {
                            str(k): v for k, v in self.tracker.player_teams.items()
                        },
                        "events": self.tracker.events,
                        "sport": self.selected_sport.get(),
                    }
                    with open(file_path, "w") as f:
                        json.dump(data, f, indent=2)
                elif export_format == "excel":
                    # Export summary to Excel
                    try:
                        import pandas as pd

                        # Create player stats DataFrame
                        player_stats = []
                        for player_id in self.tracker.players:
                            stats = self.tracker.get_player_stats(player_id)
                            if stats:
                                player_stats.append(
                                    {
                                        "Player ID": player_id,
                                        "Team": (
                                            stats["team"] + 1
                                            if stats["team"] != -1
                                            else "Unknown"
                                        ),
                                        "Distance (m)": stats["total_distance"],
                                        "Avg Speed (m/s)": stats["avg_speed"],
                                        "Max Speed (m/s)": stats["max_speed"],
                                        "Positions Tracked": stats["position_count"],
                                    }
                                )

                        # Create events DataFrame
                        events = []
                        for event in self.tracker.events:
                            frame = event.get("frame", 0)
                            time_seconds = (
                                frame / self.tracker.frame_rate
                                if self.tracker.frame_rate
                                else 0
                            )
                            minutes = int(time_seconds // 60)
                            seconds = int(time_seconds % 60)
                            time_str = f"{minutes:02d}:{seconds:02d}"

                            events.append(
                                {
                                    "Time": time_str,
                                    "Action": event.get("action", "Unknown"),
                                    "Team": (
                                        event.get("team", -1) + 1
                                        if event.get("team", -1) != -1
                                        else "Unknown"
                                    ),
                                    "Player": event.get("player", "Unknown"),
                                }
                            )

                        # Create Excel writer
                        with pd.ExcelWriter(file_path) as writer:
                            pd.DataFrame(player_stats).to_excel(
                                writer, sheet_name="Player Stats", index=False
                            )
                            pd.DataFrame(events).to_excel(
                                writer, sheet_name="Events", index=False
                            )

                            # Add possession sheet if available
                            if self.tracker.possession_data:
                                team_frames = defaultdict(int)
                                for frame, data in self.tracker.possession_data.items():
                                    team = data.get("team", -1)
                                    if team != -1:
                                        team_frames[team] += 1

                                total_frames = sum(team_frames.values())

                                if total_frames > 0:
                                    possession_data = []
                                    for team, frames in team_frames.items():
                                        percentage = (frames / total_frames) * 100
                                        possession_data.append(
                                            {
                                                "Team": team + 1,
                                                "Frames": frames,
                                                "Percentage": percentage,
                                            }
                                        )

                                    pd.DataFrame(possession_data).to_excel(
                                        writer, sheet_name="Possession", index=False
                                    )
                    except ImportError:
                        messagebox.showerror(
                            "Error",
                            "Excel export requires pandas and openpyxl packages.",
                        )
                        return

                self.status_var.set(f"Exported data to {os.path.basename(file_path)}")
                messagebox.showinfo(
                    "Success", f"Data exported successfully as {export_format.upper()}!"
                )
            except Exception as e:
                messagebox.showerror("Error", f"Error exporting data: {str(e)}")

        ttk.Button(button_frame, text="Export", command=on_export).pack(
            side=tk.LEFT, padx=10
        )
        ttk.Button(button_frame, text="Cancel", command=dialog.destroy).pack(
            side=tk.LEFT, padx=10
        )

    def generate_report(self):
        """Generate a comprehensive analysis report"""
        if not hasattr(self, "tracker") or not self.tracker:
            messagebox.showerror(
                "Error", "No tracking data available for report generation."
            )
            return

        # Ask for save location
        file_path = filedialog.asksaveasfilename(
            title="Save Analysis Report",
            defaultextension=".html",
            filetypes=[("HTML files", "*.html"), ("All files", "*.*")],
            initialdir=os.path.join(CONFIG["output_dir"], "reports"),
        )

        if not file_path:
            return

        try:
            # Create report
            self.status_var.set("Generating report...")
            sport = self.selected_sport.get()

            # Create progress dialog
            progress_dialog = tk.Toplevel(self.root)
            progress_dialog.title("Generating Report")
            progress_dialog.geometry("300x100")
            progress_dialog.transient(self.root)
            progress_dialog.grab_set()

            ttk.Label(progress_dialog, text="Generating analysis report...").pack(
                pady=(20, 10)
            )

            progress_var = tk.DoubleVar()
            progress_bar = ttk.Progressbar(
                progress_dialog, variable=progress_var, mode="indeterminate"
            )
            progress_bar.pack(fill=tk.X, padx=20, pady=10)
            progress_bar.start()

            # Generate report in a thread
            def report_thread():
                try:
                    success = self.visualizer.create_analysis_report(
                        self.analyzer.games.get("current_game", {}), file_path, sport
                    )

                    # Close dialog and update UI
                    progress_bar.stop()
                    progress_dialog.destroy()

                    if success:
                        self.status_var.set(
                            f"Generated report: {os.path.basename(file_path)}"
                        )

                        # Ask if user wants to open the report
                        if messagebox.askyesno(
                            "Report Generated",
                            "Report generated successfully! Open it now?",
                        ):
                            webbrowser.open(f"file://{os.path.abspath(file_path)}")
                    else:
                        messagebox.showerror("Error", "Failed to generate report.")
                except Exception as e:
                    progress_dialog.destroy()
                    messagebox.showerror("Error", f"Error generating report: {str(e)}")

            threading.Thread(target=report_thread, daemon=True).start()

        except Exception as e:
            messagebox.showerror("Error", f"Error generating report: {str(e)}")

    def show_player_stats(self):
        """Show detailed player statistics"""
        if not hasattr(self, "tracker") or not self.tracker:
            messagebox.showerror("Error", "No tracking data available.")
            return

        # Create dialog
        dialog = tk.Toplevel(self.root)
        dialog.title("Player Statistics")
        dialog.geometry("800x600")
        dialog.transient(self.root)

        # Get list of players
        player_ids = list(self.tracker.players.keys())

        if not player_ids:
            ttk.Label(dialog, text="No player tracking data available.").pack(pady=50)
            return

        # Create main frame
        main_frame = ttk.Frame(dialog, padding=10)
        main_frame.pack(fill=tk.BOTH, expand=True)

        # Player selection
        selection_frame = ttk.Frame(main_frame)
        selection_frame.pack(fill=tk.X, pady=(0, 10))

        ttk.Label(selection_frame, text="Select Player:").pack(side=tk.LEFT, padx=5)

        player_var = tk.StringVar()
        player_combo = ttk.Combobox(
            selection_frame,
            textvariable=player_var,
            values=[f"Player {pid}" for pid in player_ids],
            state="readonly",
            width=20,
        )
        player_combo.pack(side=tk.LEFT, padx=5)
        if player_ids:
            player_combo.current(0)

        # Stats display area with notebook tabs
        notebook = ttk.Notebook(main_frame)
        notebook.pack(fill=tk.BOTH, expand=True)

        # Overview tab
        overview_tab = ttk.Frame(notebook)
        notebook.add(overview_tab, text="Overview")

        overview_frame = ttk.Frame(overview_tab, padding=10)
        overview_frame.pack(fill=tk.BOTH, expand=True)

        # Speed tab
        speed_tab = ttk.Frame(notebook)
        notebook.add(speed_tab, text="Speed Profile")

        speed_frame = ttk.Frame(speed_tab, padding=10)
        speed_frame.pack(fill=tk.BOTH, expand=True)

        # Create a Figure for the plot
        speed_fig = Figure(figsize=(8, 4), dpi=100)
        speed_canvas = FigureCanvasTkAgg(speed_fig, master=speed_frame)
        speed_canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        # Trajectory tab
        trajectory_tab = ttk.Frame(notebook)
        notebook.add(trajectory_tab, text="Trajectory")

        trajectory_frame = ttk.Frame(trajectory_tab, padding=10)
        trajectory_frame.pack(fill=tk.BOTH, expand=True)

        # Create a Figure for the trajectory plot
        traj_fig = Figure(figsize=(8, 6), dpi=100)
        traj_canvas = FigureCanvasTkAgg(traj_fig, master=trajectory_frame)
        traj_canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        # Heatmap tab
        heatmap_tab = ttk.Frame(notebook)
        notebook.add(heatmap_tab, text="Heatmap")

        heatmap_frame = ttk.Frame(heatmap_tab, padding=10)
        heatmap_frame.pack(fill=tk.BOTH, expand=True)

        # Create a Figure for the heatmap plot
        heatmap_fig = Figure(figsize=(8, 6), dpi=100)
        heatmap_canvas = FigureCanvasTkAgg(heatmap_fig, master=heatmap_frame)
        heatmap_canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        # Actions tab
        actions_tab = ttk.Frame(notebook)
        notebook.add(actions_tab, text="Actions")

        actions_frame = ttk.Frame(actions_tab, padding=10)
        actions_frame.pack(fill=tk.BOTH, expand=True)

        # Create a Figure for the actions plot
        actions_fig = Figure(figsize=(8, 4), dpi=100)
        actions_canvas = FigureCanvasTkAgg(actions_fig, master=actions_frame)
        actions_canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        # Function to update stats display
        def update_player_display(*args):
            try:
                # Clear previous content in overview tab
                for widget in overview_frame.winfo_children():
                    widget.destroy()

                # Get selected player ID
                selected = player_var.get()
                if not selected:
                    return

                player_id = int(selected.split(" ")[1])

                # Get player stats
                stats = self.tracker.get_player_stats(player_id)
                if not stats:
                    ttk.Label(
                        overview_frame, text="No statistics available for this player."
                    ).pack(pady=50)
                    return

                # Create overview content
                # Left column for basic stats
                stats_frame = ttk.Frame(overview_frame)
                stats_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=10)

                # Player info header
                team = stats["team"]
                team_str = f"Team {team+1}" if team != -1 else "Unknown"

                header_frame = ttk.Frame(stats_frame)
                header_frame.pack(fill=tk.X, pady=(0, 20))

                ttk.Label(
                    header_frame, text=f"Player {player_id}", font=("Arial", 16, "bold")
                ).pack(side=tk.LEFT)
                team_label = ttk.Label(
                    header_frame, text=f"({team_str})", font=("Arial", 12)
                )
                team_label.pack(side=tk.LEFT, padx=10)

                # Change team label color based on team
                if team == 0:
                    team_label.configure(foreground="red")
                elif team == 1:
                    team_label.configure(foreground="blue")

                # Display basic stats
                basic_stats_frame = ttk.LabelFrame(stats_frame, text="Basic Statistics")
                basic_stats_frame.pack(fill=tk.X, pady=10)

                stats_grid = ttk.Frame(basic_stats_frame)
                stats_grid.pack(fill=tk.X, padx=10, pady=10)

                # Row 1
                ttk.Label(stats_grid, text="Total Distance:").grid(
                    row=0, column=0, sticky=tk.W, pady=5
                )
                ttk.Label(stats_grid, text=f"{stats['total_distance']:.2f} m").grid(
                    row=0, column=1, sticky=tk.W, pady=5
                )

                ttk.Label(stats_grid, text="Avg Speed:").grid(
                    row=1, column=0, sticky=tk.W, pady=5
                )
                ttk.Label(stats_grid, text=f"{stats['avg_speed']:.2f} m/s").grid(
                    row=1, column=1, sticky=tk.W, pady=5
                )

                ttk.Label(stats_grid, text="Max Speed:").grid(
                    row=2, column=0, sticky=tk.W, pady=5
                )
                ttk.Label(stats_grid, text=f"{stats['max_speed']:.2f} m/s").grid(
                    row=2, column=1, sticky=tk.W, pady=5
                )

                ttk.Label(stats_grid, text="Data Points:").grid(
                    row=3, column=0, sticky=tk.W, pady=5
                )
                ttk.Label(stats_grid, text=f"{stats['position_count']}").grid(
                    row=3, column=1, sticky=tk.W, pady=5
                )

                # Display zone percentages
                if "zone_percentages" in stats and stats["zone_percentages"]:
                    zone_frame = ttk.LabelFrame(stats_frame, text="Zone Distribution")
                    zone_frame.pack(fill=tk.X, pady=10)

                    for i, (zone, percentage) in enumerate(
                        stats["zone_percentages"].items()
                    ):
                        zone_row = ttk.Frame(zone_frame)
                        zone_row.pack(fill=tk.X, padx=10, pady=2)

                        ttk.Label(zone_row, text=f"{zone}:").pack(side=tk.LEFT)
                        ttk.Label(zone_row, text=f"{percentage:.1f}%").pack(
                            side=tk.RIGHT
                        )

                        # Progress bar for visual representation
                        bar_frame = ttk.Frame(zone_row, height=15)
                        bar_frame.pack(fill=tk.X, pady=2, padx=5, expand=True)
                        bar_canvas = tk.Canvas(
                            bar_frame, height=15, bg="white", highlightthickness=0
                        )
                        bar_canvas.pack(fill=tk.X, expand=True)

                        # Draw bar
                        bar_width = (percentage / 100) * bar_canvas.winfo_reqwidth()
                        bar_canvas.create_rectangle(
                            0, 0, bar_width, 15, fill="blue", outline=""
                        )

                # Display action counts
                if "action_counts" in stats and stats["action_counts"]:
                    action_frame = ttk.LabelFrame(stats_frame, text="Actions")
                    action_frame.pack(fill=tk.X, pady=10)

                    for action, count in stats["action_counts"].items():
                        action_row = ttk.Frame(action_frame)
                        action_row.pack(fill=tk.X, padx=10, pady=2)

                        ttk.Label(action_row, text=f"{action}:").pack(side=tk.LEFT)
                        ttk.Label(action_row, text=f"{count}").pack(side=tk.RIGHT)

                # Update speed profile tab
                speed_fig.clear()
                ax = speed_fig.add_subplot(111)

                speeds = self.tracker.player_speeds.get(player_id, [])
                if speeds:
                    # Convert frame indices to time in seconds
                    time_points = [
                        i / self.tracker.frame_rate for i in range(len(speeds))
                    ]

                    # Plot speed
                    ax.plot(time_points, speeds, "-b", label="Speed")

                    # Add average line
                    avg_speed = sum(speeds) / len(speeds)
                    ax.axhline(
                        y=avg_speed,
                        color="r",
                        linestyle="--",
                        label=f"Avg: {avg_speed:.2f} m/s",
                    )

                    # Find and mark maximum speed
                    max_speed = max(speeds)
                    max_idx = speeds.index(max_speed)
                    max_time = time_points[max_idx]
                    ax.plot(
                        max_time, max_speed, "ro", label=f"Max: {max_speed:.2f} m/s"
                    )

                    # Add speed thresholds
                    thresholds = [
                        (2.0, "Walking"),
                        (4.0, "Jogging"),
                        (7.0, "Running"),
                        (9.0, "Sprinting"),
                    ]
                    colors = ["green", "orange", "red", "purple"]

                    for i, (threshold, label) in enumerate(thresholds):
                        ax.axhline(
                            y=threshold,
                            color=colors[i],
                            linestyle=":",
                            alpha=0.7,
                            label=f"{label} ({threshold} m/s)",
                        )

                    ax.set_xlabel("Time (seconds)")
                    ax.set_ylabel("Speed (m/s)")
                    ax.set_title(f"Player {player_id} Speed Profile")
                    ax.grid(True, alpha=0.3)
                    ax.legend()
                else:
                    ax.text(
                        0.5,
                        0.5,
                        "No speed data available",
                        horizontalalignment="center",
                        verticalalignment="center",
                        transform=ax.transAxes,
                    )

                speed_canvas.draw()

                # Update trajectory tab
                traj_fig.clear()
                ax = traj_fig.add_subplot(111)

                positions = self.tracker.players.get(player_id, [])
                if positions:
                    # Extract x and y coordinates
                    x_coords, y_coords = zip(*positions)

                    # Create colormap for time progression
                    norm = Normalize(vmin=0, vmax=len(positions) - 1)
                    colors = [plt.cm.viridis(norm(i)) for i in range(len(positions))]

                    # Plot the trajectory
                    for i in range(len(positions) - 1):
                        ax.plot(
                            [x_coords[i], x_coords[i + 1]],
                            [y_coords[i], y_coords[i + 1]],
                            color=colors[i],
                            linewidth=2,
                        )

                    # Mark start and end points
                    ax.plot(
                        x_coords[0], y_coords[0], "go", markersize=10, label="Start"
                    )
                    ax.plot(
                        x_coords[-1], y_coords[-1], "ro", markersize=10, label="End"
                    )

                    # Add field dimensions based on sport
                    sport = self.selected_sport.get()
                    field_width, field_height = CONFIG["sports"][sport][
                        "field_dimensions"
                    ]

                    # Draw field rectangle
                    rect = plt.Rectangle(
                        (0, 0),
                        field_width,
                        field_height,
                        linewidth=2,
                        edgecolor="green",
                        facecolor="none",
                    )
                    ax.add_patch(rect)

                    # Draw center line and circle for football/soccer
                    if sport == "football":
                        # Center line
                        ax.plot(
                            [field_width / 2, field_width / 2],
                            [0, field_height],
                            "white",
                            linestyle="-",
                        )

                        # Center circle
                        circle = plt.Circle(
                            (field_width / 2, field_height / 2),
                            9.15,
                            edgecolor="white",
                            facecolor="none",
                        )
                        ax.add_patch(circle)

                    ax.set_xlabel("X Position (meters)")
                    ax.set_ylabel("Y Position (meters)")
                    ax.set_title(f"Player {player_id} Trajectory")
                    ax.grid(True, alpha=0.3)
                    ax.legend()

                    # Set axis limits with some padding
                    ax.set_xlim(-5, field_width + 5)
                    ax.set_ylim(-5, field_height + 5)
                else:
                    ax.text(
                        0.5,
                        0.5,
                        "No trajectory data available",
                        horizontalalignment="center",
                        verticalalignment="center",
                        transform=ax.transAxes,
                    )

                traj_canvas.draw()

                # Update heatmap tab
                heatmap_fig.clear()
                ax = heatmap_fig.add_subplot(111)

                if "heatmap_data" in stats and stats["heatmap_data"] is not None:
                    heatmap_data = stats["heatmap_data"]

                    # Get field dimensions for the sport
                    sport = self.selected_sport.get()
                    field_width, field_height = CONFIG["sports"][sport][
                        "field_dimensions"
                    ]

                    # Plot heatmap
                    extent = [0, field_width, 0, field_height]
                    im = ax.imshow(
                        heatmap_data,
                        cmap="viridis",
                        origin="lower",
                        extent=extent,
                        aspect="auto",
                    )

                    # Add colorbar
                    cbar = heatmap_fig.colorbar(im, ax=ax)
                    cbar.set_label("Density")

                    # Add field dimensions based on sport
                    if sport == "football":
                        # Center line
                        ax.plot(
                            [field_width / 2, field_width / 2],
                            [0, field_height],
                            "white",
                            linestyle="-",
                        )

                        # Center circle
                        circle = plt.Circle(
                            (field_width / 2, field_height / 2),
                            9.15,
                            edgecolor="white",
                            facecolor="none",
                        )
                        ax.add_patch(circle)

                        # Penalty areas
                        rect1 = plt.Rectangle(
                            (0, (field_height - 40.3) / 2),
                            16.5,
                            40.3,
                            linewidth=2,
                            edgecolor="white",
                            facecolor="none",
                        )
                        ax.add_patch(rect1)

                        rect2 = plt.Rectangle(
                            (field_width - 16.5, (field_height - 40.3) / 2),
                            16.5,
                            40.3,
                            linewidth=2,
                            edgecolor="white",
                            facecolor="none",
                        )
                        ax.add_patch(rect2)

                    ax.set_xlabel("X Position (meters)")
                    ax.set_ylabel("Y Position (meters)")
                    ax.set_title(f"Player {player_id} Heatmap")
                else:
                    ax.text(
                        0.5,
                        0.5,
                        "No heatmap data available",
                        horizontalalignment="center",
                        verticalalignment="center",
                        transform=ax.transAxes,
                    )

                heatmap_canvas.draw()

                # Update actions tab
                actions_fig.clear()
                ax = actions_fig.add_subplot(111)

                if "action_counts" in stats and stats["action_counts"]:
                    actions = list(stats["action_counts"].keys())
                    counts = list(stats["action_counts"].values())

                    bars = ax.bar(actions, counts, color="skyblue")

                    # Add count labels above bars
                    for bar in bars:
                        height = bar.get_height()
                        ax.text(
                            bar.get_x() + bar.get_width() / 2.0,
                            height + 0.1,
                            f"{int(height)}",
                            ha="center",
                            va="bottom",
                        )

                    ax.set_xlabel("Actions")
                    ax.set_ylabel("Count")
                    ax.set_title(f"Player {player_id} Actions")
                    ax.set_ylim(0, max(counts) * 1.2 if counts else 10)

                    # Rotate x labels if needed
                    if len(actions) > 3:
                        plt.setp(ax.get_xticklabels(), rotation=45, ha="right")
                else:
                    ax.text(
                        0.5,
                        0.5,
                        "No action data available",
                        horizontalalignment="center",
                        verticalalignment="center",
                        transform=ax.transAxes,
                    )

                actions_canvas.draw()

            except Exception as e:
                messagebox.showerror(
                    "Error", f"Error displaying player stats: {str(e)}"
                )

        # Update display when player selection changes
        player_var.trace("w", update_player_display)

        # Initial update
        update_player_display()

    def show_team_analysis(self):
        """Show team analysis"""
        if not hasattr(self, "tracker") or not self.tracker:
            messagebox.showerror("Error", "No tracking data available.")
            return

        # Check if we have team data
        team_players = defaultdict(list)
        for player_id, team in self.tracker.player_teams.items():
            if team != -1:
                team_players[team].append(player_id)

        if not team_players:
            messagebox.showerror(
                "Error",
                "No team data available. Enable team detection during tracking.",
            )
            return

        # Create dialog
        dialog = tk.Toplevel(self.root)
        dialog.title("Team Analysis")
        dialog.geometry("800x600")
        dialog.transient(self.root)

        # Create main frame
        main_frame = ttk.Frame(dialog, padding=10)
        main_frame.pack(fill=tk.BOTH, expand=True)

        # Team selection
        selection_frame = ttk.Frame(main_frame)
        selection_frame.pack(fill=tk.X, pady=(0, 10))

        ttk.Label(selection_frame, text="Select Team:").pack(side=tk.LEFT, padx=5)

        team_var = tk.StringVar()
        team_combo = ttk.Combobox(
            selection_frame,
            textvariable=team_var,
            values=[f"Team {team+1}" for team in team_players.keys()],
            state="readonly",
            width=20,
        )
        team_combo.pack(side=tk.LEFT, padx=5)
        team_combo.current(0)

        # Stats display area with notebook tabs
        notebook = ttk.Notebook(main_frame)
        notebook.pack(fill=tk.BOTH, expand=True)

        # Overview tab
        overview_tab = ttk.Frame(notebook)
        notebook.add(overview_tab, text="Overview")

        overview_frame = ttk.Frame(overview_tab, padding=10)
        overview_frame.pack(fill=tk.BOTH, expand=True)

        # Heatmap tab
        heatmap_tab = ttk.Frame(notebook)
        notebook.add(heatmap_tab, text="Heatmap")

        heatmap_frame = ttk.Frame(heatmap_tab, padding=10)
        heatmap_frame.pack(fill=tk.BOTH, expand=True)

        # Create a Figure for the heatmap plot
        heatmap_fig = Figure(figsize=(8, 6), dpi=100)
        heatmap_canvas = FigureCanvasTkAgg(heatmap_fig, master=heatmap_frame)
        heatmap_canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        # Possession tab
        possession_tab = ttk.Frame(notebook)
        notebook.add(possession_tab, text="Possession")

        possession_frame = ttk.Frame(possession_tab, padding=10)
        possession_frame.pack(fill=tk.BOTH, expand=True)

        # Create a Figure for the possession plot
        possession_fig = Figure(figsize=(8, 4), dpi=100)
        possession_canvas = FigureCanvasTkAgg(possession_fig, master=possession_frame)
        possession_canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        # Formation tab
        formation_tab = ttk.Frame(notebook)
        notebook.add(formation_tab, text="Formation")

        formation_frame = ttk.Frame(formation_tab, padding=10)
        formation_frame.pack(fill=tk.BOTH, expand=True)

        # Create a Figure for the formation plot
        formation_fig = Figure(figsize=(8, 6), dpi=100)
        formation_canvas = FigureCanvasTkAgg(formation_fig, master=formation_frame)
        formation_canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        # Function to update team display
        def update_team_display(*args):
            try:
                # Clear previous content in overview tab
                for widget in overview_frame.winfo_children():
                    widget.destroy()

                # Get selected team
                selected = team_var.get()
                if not selected:
                    return

                team_id = int(selected.split(" ")[1]) - 1

                # Get team stats
                stats = self.tracker.get_team_stats(team_id)
                if not stats:
                    ttk.Label(
                        overview_frame, text="No statistics available for this team."
                    ).pack(pady=50)
                    return

                # Create overview content
                # Header
                header_frame = ttk.Frame(overview_frame)
                header_frame.pack(fill=tk.X, pady=(0, 20))

                team_color = "red" if team_id == 0 else "blue"
                team_label = ttk.Label(
                    header_frame,
                    text=f"Team {team_id+1}",
                    font=("Arial", 16, "bold"),
                    foreground=team_color,
                )
                team_label.pack(side=tk.LEFT)

                # Key stats in overview
                stats_grid = ttk.Frame(overview_frame)
                stats_grid.pack(fill=tk.X, pady=10)

                # Row 1
                ttk.Label(stats_grid, text="Players:", font=("Arial", 11, "bold")).grid(
                    row=0, column=0, sticky=tk.W, pady=5
                )
                ttk.Label(stats_grid, text=f"{stats['player_count']}").grid(
                    row=0, column=1, sticky=tk.W, pady=5
                )

                ttk.Label(
                    stats_grid, text="Total Distance:", font=("Arial", 11, "bold")
                ).grid(row=1, column=0, sticky=tk.W, pady=5)
                ttk.Label(stats_grid, text=f"{stats['total_distance']:.2f} m").grid(
                    row=1, column=1, sticky=tk.W, pady=5
                )

                ttk.Label(
                    stats_grid, text="Avg Speed:", font=("Arial", 11, "bold")
                ).grid(row=2, column=0, sticky=tk.W, pady=5)
                ttk.Label(stats_grid, text=f"{stats['avg_speed']:.2f} m/s").grid(
                    row=2, column=1, sticky=tk.W, pady=5
                )

                ttk.Label(
                    stats_grid, text="Max Speed:", font=("Arial", 11, "bold")
                ).grid(row=3, column=0, sticky=tk.W, pady=5)
                ttk.Label(stats_grid, text=f"{stats['max_speed']:.2f} m/s").grid(
                    row=3, column=1, sticky=tk.W, pady=5
                )

                # Possession
                if "possession_percentage" in stats:
                    ttk.Label(
                        stats_grid, text="Possession:", font=("Arial", 11, "bold")
                    ).grid(row=4, column=0, sticky=tk.W, pady=5)
                    ttk.Label(
                        stats_grid, text=f"{stats['possession_percentage']:.1f}%"
                    ).grid(row=4, column=1, sticky=tk.W, pady=5)

                # Player list
                players_frame = ttk.LabelFrame(overview_frame, text="Players")
                players_frame.pack(fill=tk.BOTH, expand=True, pady=10)

                # Create scrollable player list
                players_canvas = tk.Canvas(players_frame)
                scrollbar = ttk.Scrollbar(
                    players_frame, orient="vertical", command=players_canvas.yview
                )
                scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
                players_canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
                players_canvas.configure(yscrollcommand=scrollbar.set)

                players_content = ttk.Frame(players_canvas)
                players_canvas.create_window(
                    (0, 0), window=players_content, anchor="nw"
                )

                # Configure the canvas scrolling
                def configure_players_canvas(event):
                    players_canvas.configure(
                        scrollregion=players_canvas.bbox("all"), width=event.width
                    )

                players_content.bind("<Configure>", configure_players_canvas)

                # Add headers
                columns = ("ID", "Distance (m)", "Avg Speed (m/s)", "Max Speed (m/s)")
                header_frame = ttk.Frame(players_content)
                header_frame.pack(fill=tk.X, pady=5)

                for i, col in enumerate(columns):
                    ttk.Label(header_frame, text=col, font=("Arial", 10, "bold")).grid(
                        row=0, column=i, sticky=tk.W, padx=10
                    )

                # Add player rows
                row = 1
                for player_id in team_players[team_id]:
                    player_stats = self.tracker.get_player_stats(player_id)
                    if player_stats:
                        player_frame = ttk.Frame(players_content)
                        player_frame.pack(fill=tk.X, pady=2)

                        ttk.Label(player_frame, text=f"{player_id}").grid(
                            row=0, column=0, sticky=tk.W, padx=10
                        )
                        ttk.Label(
                            player_frame, text=f"{player_stats['total_distance']:.2f}"
                        ).grid(row=0, column=1, sticky=tk.W, padx=10)
                        ttk.Label(
                            player_frame, text=f"{player_stats['avg_speed']:.2f}"
                        ).grid(row=0, column=2, sticky=tk.W, padx=10)
                        ttk.Label(
                            player_frame, text=f"{player_stats['max_speed']:.2f}"
                        ).grid(row=0, column=3, sticky=tk.W, padx=10)

                # Update heatmap tab
                heatmap_fig.clear()
                ax = heatmap_fig.add_subplot(111)

                heatmap_data = stats.get("heatmap_data")
                if heatmap_data is not None:
                    # Get field dimensions for the sport
                    sport = self.selected_sport.get()
                    field_width, field_height = CONFIG["sports"][sport][
                        "field_dimensions"
                    ]

                    # Plot heatmap
                    extent = [0, field_width, 0, field_height]
                    im = ax.imshow(
                        heatmap_data,
                        cmap="viridis",
                        origin="lower",
                        extent=extent,
                        aspect="auto",
                    )

                    # Add colorbar
                    cbar = heatmap_fig.colorbar(im, ax=ax)
                    cbar.set_label("Density")

                    # Add field dimensions based on sport
                    if sport == "football":
                        # Center line
                        ax.plot(
                            [field_width / 2, field_width / 2],
                            [0, field_height],
                            "white",
                            linestyle="-",
                        )

                        # Center circle
                        circle = plt.Circle(
                            (field_width / 2, field_height / 2),
                            9.15,
                            edgecolor="white",
                            facecolor="none",
                        )
                        ax.add_patch(circle)

                        # Penalty areas
                        rect1 = plt.Rectangle(
                            (0, (field_height - 40.3) / 2),
                            16.5,
                            40.3,
                            linewidth=2,
                            edgecolor="white",
                            facecolor="none",
                        )
                        ax.add_patch(rect1)

                        rect2 = plt.Rectangle(
                            (field_width - 16.5, (field_height - 40.3) / 2),
                            16.5,
                            40.3,
                            linewidth=2,
                            edgecolor="white",
                            facecolor="none",
                        )
                        ax.add_patch(rect2)

                    ax.set_xlabel("X Position (meters)")
                    ax.set_ylabel("Y Position (meters)")
                    ax.set_title(f"Team {team_id+1} Heatmap")
                else:
                    ax.text(
                        0.5,
                        0.5,
                        "No heatmap data available",
                        horizontalalignment="center",
                        verticalalignment="center",
                        transform=ax.transAxes,
                    )

                heatmap_canvas.draw()

                # Update possession tab
                possession_fig.clear()

                # Get possession data
                possession_data = self.analyzer.analyze_possession("current_game")
                if possession_data:
                    # Create two subplots
                    ax1 = possession_fig.add_subplot(121)
                    ax2 = possession_fig.add_subplot(122)

                    # Plot 1: Possession pie chart
                    percentages = possession_data["possession_percentages"]
                    if percentages:
                        labels = [
                            f"Team {team+1}: {pct:.1f}%"
                            for team, pct in percentages.items()
                        ]
                        values = list(percentages.values())
                        colors = ["red", "blue"]

                        ax1.pie(
                            values,
                            labels=labels,
                            colors=colors,
                            autopct="%1.1f%%",
                            startangle=90,
                        )
                        ax1.axis(
                            "equal"
                        )  # Equal aspect ratio ensures that pie is drawn as a circle
                        ax1.set_title("Ball Possession")
                    else:
                        ax1.text(
                            0.5,
                            0.5,
                            "No possession data",
                            horizontalalignment="center",
                            verticalalignment="center",
                            transform=ax1.transAxes,
                        )

                    # Plot 2: Possession by zone
                    zone_possession = possession_data["zone_possession"]
                    if zone_possession:
                        zones = list(zone_possession.keys())
                        team_data = [
                            zone_possession[zone].get(team_id, 0) for zone in zones
                        ]
                        other_team = 1 if team_id == 0 else 0
                        other_team_data = [
                            zone_possession[zone].get(other_team, 0) for zone in zones
                        ]

                        # Calculate percentages
                        totals = [
                            team + other
                            for team, other in zip(team_data, other_team_data)
                        ]
                        team_percentages = [
                            team / total * 100 if total > 0 else 0
                            for team, total in zip(team_data, totals)
                        ]

                        x = range(len(zones))

                        ax2.bar(x, team_percentages, color=team_color, alpha=0.7)
                        ax2.set_xticks(x)
                        ax2.set_xticklabels(zones, rotation=45, ha="right")
                        ax2.set_ylabel("Possession (%)")
                        ax2.set_title("Zone Control")
                        ax2.set_ylim(0, 100)

                        # Add percentage labels
                        for i, pct in enumerate(team_percentages):
                            ax2.text(
                                i, pct + 2, f"{pct:.1f}%", ha="center", va="bottom"
                            )
                    else:
                        ax2.text(
                            0.5,
                            0.5,
                            "No zone data",
                            horizontalalignment="center",
                            verticalalignment="center",
                            transform=ax2.transAxes,
                        )
                else:
                    ax = possession_fig.add_subplot(111)
                    ax.text(
                        0.5,
                        0.5,
                        "No possession data available",
                        horizontalalignment="center",
                        verticalalignment="center",
                        transform=ax.transAxes,
                    )

                possession_fig.tight_layout()
                possession_canvas.draw()

                # Update formation tab
                formation_fig.clear()
                ax = formation_fig.add_subplot(111)

                # Get average positions for all players in the team
                positions = []
                for player_id in team_players[team_id]:
                    if (
                        player_id in self.tracker.players
                        and len(self.tracker.players[player_id]) > 0
                    ):
                        player_positions = self.tracker.players[player_id]
                        avg_pos = np.mean(player_positions, axis=0)
                        positions.append((player_id, avg_pos))

                if positions:
                    # Get field dimensions for the sport
                    sport = self.selected_sport.get()
                    field_width, field_height = CONFIG["sports"][sport][
                        "field_dimensions"
                    ]

                    # Draw field
                    rect = plt.Rectangle(
                        (0, 0),
                        field_width,
                        field_height,
                        linewidth=2,
                        edgecolor="green",
                        facecolor="lightgreen",
                        alpha=0.3,
                    )
                    ax.add_patch(rect)

                    # Draw center line and circle for football/soccer
                    if sport == "football":
                        # Center line
                        ax.plot(
                            [field_width / 2, field_width / 2],
                            [0, field_height],
                            "white",
                            linestyle="-",
                        )

                        # Center circle
                        circle = plt.Circle(
                            (field_width / 2, field_height / 2),
                            9.15,
                            edgecolor="white",
                            facecolor="none",
                        )
                        ax.add_patch(circle)

                        # Penalty areas
                        rect1 = plt.Rectangle(
                            (0, (field_height - 40.3) / 2),
                            16.5,
                            40.3,
                            linewidth=2,
                            edgecolor="white",
                            facecolor="none",
                        )
                        ax.add_patch(rect1)

                        rect2 = plt.Rectangle(
                            (field_width - 16.5, (field_height - 40.3) / 2),
                            16.5,
                            40.3,
                            linewidth=2,
                            edgecolor="white",
                            facecolor="none",
                        )
                        ax.add_patch(rect2)

                    # Plot player positions
                    for player_id, pos in positions:
                        ax.plot(pos[0], pos[1], "o", markersize=12, color=team_color)
                        ax.text(
                            pos[0],
                            pos[1] + 1,
                            str(player_id),
                            ha="center",
                            va="bottom",
                            color="white",
                            fontweight="bold",
                        )

                    # Calculate and draw connections between closest players
                    if len(positions) >= 2:
                        # Build a graph of player connections
                        G = nx.Graph()
                        for player_id, pos in positions:
                            G.add_node(player_id, pos=pos)

                        # Add edges between players (connect each player to their 2-3 closest neighbors)
                        for i, (player1, pos1) in enumerate(positions):
                            # Calculate distances to all other players
                            distances = []
                            for j, (player2, pos2) in enumerate(positions):
                                if i != j:
                                    dist = np.sqrt(
                                        (pos1[0] - pos2[0]) ** 2
                                        + (pos1[1] - pos2[1]) ** 2
                                    )
                                    distances.append((j, dist))

                            # Sort by distance and connect to closest players
                            distances.sort(key=lambda x: x[1])
                            for j, dist in distances[: min(3, len(distances))]:
                                G.add_edge(player1, positions[j][0])

                        # Draw edges
                        for u, v in G.edges():
                            u_pos = G.nodes[u]["pos"]
                            v_pos = G.nodes[v]["pos"]
                            ax.plot(
                                [u_pos[0], v_pos[0]],
                                [u_pos[1], v_pos[1]],
                                color=team_color,
                                alpha=0.5,
                                linestyle="-",
                            )

                    ax.set_xlim(-5, field_width + 5)
                    ax.set_ylim(-5, field_height + 5)
                    ax.set_xlabel("X Position (meters)")
                    ax.set_ylabel("Y Position (meters)")
                    ax.set_title(f"Team {team_id+1} Formation")
                else:
                    ax.text(
                        0.5,
                        0.5,
                        "No formation data available",
                        horizontalalignment="center",
                        verticalalignment="center",
                        transform=ax.transAxes,
                    )

                formation_canvas.draw()

            except Exception as e:
                messagebox.showerror(
                    "Error", f"Error displaying team analysis: {str(e)}"
                )

        # Update display when team selection changes
        team_var.trace("w", update_team_display)

        # Initial update
        update_team_display()

    def show_event_timeline(self):
        """Show event timeline"""
        if not hasattr(self, "tracker") or not self.tracker:
            messagebox.showerror("Error", "No tracking data available.")
            return

        # Check if we have event data
        if not self.tracker.events:
            messagebox.showerror("Error", "No event data available.")
            return

        # Create dialog
        dialog = tk.Toplevel(self.root)
        dialog.title("Event Timeline")
        dialog.geometry("800x600")
        dialog.transient(self.root)

        # Create main frame
        main_frame = ttk.Frame(dialog, padding=10)
        main_frame.pack(fill=tk.BOTH, expand=True)

        # Add filter controls
        filter_frame = ttk.LabelFrame(main_frame, text="Filters")
        filter_frame.pack(fill=tk.X, pady=(0, 10))

        filter_controls = ttk.Frame(filter_frame)
        filter_controls.pack(fill=tk.X, padx=10, pady=10)

        # Team filter
        ttk.Label(filter_controls, text="Team:").grid(
            row=0, column=0, sticky=tk.W, padx=5
        )

        team_var = tk.StringVar(value="All")
        team_combo = ttk.Combobox(
            filter_controls,
            textvariable=team_var,
            width=15,
            state="readonly",
            values=["All", "Team 1", "Team 2"],
        )
        team_combo.grid(row=0, column=1, sticky=tk.W, padx=5)

        # Action filter
        ttk.Label(filter_controls, text="Action:").grid(
            row=0, column=2, sticky=tk.W, padx=5
        )

        # Get unique actions
        actions = set()
        for event in self.tracker.events:
            actions.add(event.get("action", "Unknown"))

        action_var = tk.StringVar(value="All")
        action_combo = ttk.Combobox(
            filter_controls,
            textvariable=action_var,
            width=15,
            state="readonly",
            values=["All"] + sorted(list(actions)),
        )
        action_combo.grid(row=0, column=3, sticky=tk.W, padx=5)

        # Player filter
        ttk.Label(filter_controls, text="Player:").grid(
            row=0, column=4, sticky=tk.W, padx=5
        )

        # Get unique players
        players = set()
        for event in self.tracker.events:
            player = event.get("player")
            if player is not None:
                players.add(player)

        player_var = tk.StringVar(value="All")
        player_combo = ttk.Combobox(
            filter_controls,
            textvariable=player_var,
            width=15,
            state="readonly",
            values=["All"] + [f"Player {p}" for p in sorted(list(players))],
        )
        player_combo.grid(row=0, column=5, sticky=tk.W, padx=5)

        # Timeline visualization
        timeline_frame = ttk.Frame(main_frame)
        timeline_frame.pack(fill=tk.BOTH, expand=True, pady=10)

        # Create canvas for timeline
        timeline_fig = Figure(figsize=(8, 4), dpi=100)
        timeline_canvas = FigureCanvasTkAgg(timeline_fig, master=timeline_frame)
        timeline_canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        # Event list
        events_frame = ttk.LabelFrame(main_frame, text="Events")
        events_frame.pack(fill=tk.BOTH, expand=True, pady=10)

        # Create a treeview for events
        columns = ("Time", "Action", "Team", "Player", "Position")
        events_tree = ttk.Treeview(events_frame, columns=columns, show="headings")

        # Define headings
        for col in columns:
            events_tree.heading(col, text=col)
            events_tree.column(col, width=100)

        # Add scrollbar
        scrollbar = ttk.Scrollbar(
            events_frame, orient=tk.VERTICAL, command=events_tree.yview
        )
        events_tree.configure(yscrollcommand=scrollbar.set)

        # Pack tree and scrollbar
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        events_tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        # Function to update the display based on filters
        def update_display(*args):
            # Get filter values
            team_filter = team_var.get()
            action_filter = action_var.get()
            player_filter = player_var.get()

            # Apply filters
            filtered_events = []
            for event in self.tracker.events:
                # Team filter
                if team_filter != "All":
                    team = event.get("team", -1)
                    if team == -1 or team + 1 != int(team_filter.split(" ")[1]):
                        continue

                # Action filter
                if action_filter != "All" and event.get("action") != action_filter:
                    continue

                # Player filter
                if player_filter != "All":
                    player = event.get("player")
                    if player is None or f"Player {player}" != player_filter:
                        continue

                filtered_events.append(event)

            # Sort by time
            filtered_events.sort(key=lambda e: e.get("frame", 0))

            # Update timeline visualization
            timeline_fig.clear()
            ax = timeline_fig.add_subplot(111)

            if filtered_events:
                # Extract data for timeline
                frames = [e.get("frame", 0) for e in filtered_events]
                times = [e.get("time", 0) for e in filtered_events]
                actions = [e.get("action", "Unknown") for e in filtered_events]
                teams = [e.get("team", -1) for e in filtered_events]

                # Get unique actions for y-axis
                unique_actions = sorted(set(actions))

                # Create a mapping from action to y position
                action_to_y = {action: i for i, action in enumerate(unique_actions)}

                # Plot points on timeline
                for i, (time, action, team) in enumerate(zip(times, actions, teams)):
                    color = "red" if team == 0 else "blue" if team == 1 else "gray"
                    ax.scatter(time, action_to_y[action], color=color, s=100, alpha=0.7)

                # Add event labels
                for i, (time, action) in enumerate(zip(times, actions)):
                    if i % 3 == 0:  # Add labels sparingly to avoid clutter
                        ax.text(
                            time,
                            action_to_y[action] + 0.1,
                            action,
                            ha="center",
                            va="bottom",
                            fontsize=8,
                            alpha=0.7,
                        )

                # Set y-axis ticks and labels
                ax.set_yticks(range(len(unique_actions)))
                ax.set_yticklabels(unique_actions)

                # Set x-axis as time
                ax.set_xlabel("Time (seconds)")

                # Add grid
                ax.grid(True, alpha=0.3)

                ax.set_title("Event Timeline")
            else:
                ax.text(
                    0.5,
                    0.5,
                    "No events match the current filters",
                    horizontalalignment="center",
                    verticalalignment="center",
                    transform=ax.transAxes,
                )

            timeline_fig.tight_layout()
            timeline_canvas.draw()

            # Update event list
            events_tree.delete(*events_tree.get_children())

            for event in filtered_events:
                # Format time
                frame = event.get("frame", 0)
                time_seconds = (
                    frame / self.tracker.frame_rate if self.tracker.frame_rate else 0
                )
                minutes = int(time_seconds // 60)
                seconds = int(time_seconds % 60)
                time_str = f"{minutes:02d}:{seconds:02d}"

                # Format team
                team = event.get("team", -1)
                team_str = f"Team {team+1}" if team != -1 else "Unknown"

                # Format player
                player = event.get("player")
                player_str = f"Player {player}" if player is not None else "Unknown"

                # Format position
                position = event.get("position")
                position_str = (
                    f"({position[0]:.1f}, {position[1]:.1f})" if position else "Unknown"
                )

                # Add to treeview
                events_tree.insert(
                    "",
                    "end",
                    values=(
                        time_str,
                        event.get("action", "Unknown"),
                        team_str,
                        player_str,
                        position_str,
                    ),
                )

        # Update display when filters change
        team_var.trace("w", update_display)
        action_var.trace("w", update_display)
        player_var.trace("w", update_display)

        # Initial update
        update_display()

    def show_heatmaps(self):
        """Show heatmap visualization"""
        if not hasattr(self, "tracker") or not self.tracker:
            messagebox.showerror("Error", "No tracking data available.")
            return

        # Create dialog
        dialog = tk.Toplevel(self.root)
        dialog.title("Heatmap Visualization")
        dialog.geometry("800x700")
        dialog.transient(self.root)

        # Create main frame
        main_frame = ttk.Frame(dialog, padding=10)
        main_frame.pack(fill=tk.BOTH, expand=True)

        # Control panel
        control_frame = ttk.Frame(main_frame)
        control_frame.pack(fill=tk.X, pady=(0, 10))

        # Select visualization type
        type_frame = ttk.LabelFrame(control_frame, text="Visualization Type")
        type_frame.pack(side=tk.LEFT, fill=tk.Y, padx=5)

        type_var = tk.StringVar(value="team")
        ttk.Radiobutton(
            type_frame, text="Team Heatmap", variable=type_var, value="team"
        ).pack(anchor=tk.W, padx=10, pady=2)
        ttk.Radiobutton(
            type_frame, text="Player Heatmap", variable=type_var, value="player"
        ).pack(anchor=tk.W, padx=10, pady=2)
        ttk.Radiobutton(
            type_frame, text="Zone Analysis", variable=type_var, value="zone"
        ).pack(anchor=tk.W, padx=10, pady=2)

        # Selection panel (changes based on type)
        selection_frame = ttk.LabelFrame(control_frame, text="Selection")
        selection_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=5)

        # Team selection
        team_var = tk.StringVar()
        team_frame = ttk.Frame(selection_frame)
        team_combo = ttk.Combobox(
            team_frame,
            textvariable=team_var,
            width=15,
            state="readonly",
            values=["Team 1", "Team 2"],
        )

        # Player selection
        player_var = tk.StringVar()
        player_frame = ttk.Frame(selection_frame)

        # Get list of players
        player_ids = list(self.tracker.players.keys())
        player_combo = ttk.Combobox(
            player_frame,
            textvariable=player_var,
            width=15,
            state="readonly",
            values=[f"Player {pid}" for pid in player_ids],
        )

        # Zone selection
        zone_var = tk.StringVar()
        zone_frame = ttk.Frame(selection_frame)

        # Get list of zones
        sport = self.selected_sport.get()
        zones = list(CONFIG["sports"][sport]["zones"].keys())
        zone_combo = ttk.Combobox(
            zone_frame, textvariable=zone_var, width=15, state="readonly", values=zones
        )

        # Display options
        options_frame = ttk.LabelFrame(control_frame, text="Options")
        options_frame.pack(side=tk.LEFT, fill=tk.Y, padx=5)

        # Colormap selection
        colormap_frame = ttk.Frame(options_frame)
        colormap_frame.pack(fill=tk.X, padx=10, pady=5)

        ttk.Label(colormap_frame, text="Colormap:").pack(side=tk.LEFT)

        colormap_var = tk.StringVar(value="viridis")
        colormap_combo = ttk.Combobox(
            colormap_frame,
            textvariable=colormap_var,
            width=15,
            state="readonly",
            values=[
                "viridis",
                "plasma",
                "magma",
                "inferno",
                "cividis",
                "Reds",
                "Blues",
                "Greens",
                "YlOrRd",
                "coolwarm",
            ],
        )
        colormap_combo.pack(side=tk.LEFT, padx=5)

        # Show field markings
        field_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(
            options_frame, text="Show Field Markings", variable=field_var
        ).pack(anchor=tk.W, padx=10, pady=5)

        # Normalize data
        normalize_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(
            options_frame, text="Normalize Data", variable=normalize_var
        ).pack(anchor=tk.W, padx=10, pady=5)

        # Visualization area
        vis_frame = ttk.Frame(main_frame)
        vis_frame.pack(fill=tk.BOTH, expand=True, pady=10)

        # Create matplotlib figure
        fig = Figure(figsize=(8, 6), dpi=100)
        canvas = FigureCanvasTkAgg(fig, master=vis_frame)
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        # Toolbar
        from matplotlib.backends.backend_tkagg import NavigationToolbar2Tk

        toolbar = NavigationToolbar2Tk(canvas, vis_frame)
        toolbar.update()

        # Function to update the selection frame based on type
        def update_selection_frame(*args):
            # Clear current content
            for widget in selection_frame.winfo_children():
                widget.pack_forget()

            vis_type = type_var.get()

            if vis_type == "team":
                team_frame.pack(fill=tk.X, padx=10, pady=10)
                ttk.Label(team_frame, text="Select Team:").pack(side=tk.LEFT)
                team_combo.pack(side=tk.LEFT, padx=5)
                team_combo.current(0)
            elif vis_type == "player":
                player_frame.pack(fill=tk.X, padx=10, pady=10)
                ttk.Label(player_frame, text="Select Player:").pack(side=tk.LEFT)
                player_combo.pack(side=tk.LEFT, padx=5)
                if player_ids:
                    player_combo.current(0)
            elif vis_type == "zone":
                zone_frame.pack(fill=tk.X, padx=10, pady=10)
                ttk.Label(zone_frame, text="Select Zone:").pack(side=tk.LEFT)
                zone_combo.pack(side=tk.LEFT, padx=5)
                if zones:
                    zone_combo.current(0)

            # Update visualization
            update_visualization()

        # Function to update visualization
        def update_visualization(*args):
            try:
                vis_type = type_var.get()
                colormap = colormap_var.get()
                show_field = field_var.get()
                normalize = normalize_var.get()

                # Clear figure
                fig.clear()
                ax = fig.add_subplot(111)

                # Get field dimensions for the sport
                sport = self.selected_sport.get()
                field_width, field_height = CONFIG["sports"][sport]["field_dimensions"]

                # Draw field if requested
                if show_field:
                    # Field background
                    rect = plt.Rectangle(
                        (0, 0),
                        field_width,
                        field_height,
                        linewidth=2,
                        edgecolor="green",
                        facecolor="lightgreen",
                        alpha=0.3,
                    )
                    ax.add_patch(rect)

                    # Add field markings based on sport
                    if sport == "football":
                        # Center line
                        ax.plot(
                            [field_width / 2, field_width / 2],
                            [0, field_height],
                            "white",
                            linestyle="-",
                        )

                        # Center circle
                        circle = plt.Circle(
                            (field_width / 2, field_height / 2),
                            9.15,
                            edgecolor="white",
                            facecolor="none",
                        )
                        ax.add_patch(circle)

                        # Penalty areas
                        rect1 = plt.Rectangle(
                            (0, (field_height - 40.3) / 2),
                            16.5,
                            40.3,
                            linewidth=2,
                            edgecolor="white",
                            facecolor="none",
                        )
                        ax.add_patch(rect1)

                        rect2 = plt.Rectangle(
                            (field_width - 16.5, (field_height - 40.3) / 2),
                            16.5,
                            40.3,
                            linewidth=2,
                            edgecolor="white",
                            facecolor="none",
                        )
                        ax.add_patch(rect2)

                        # Goal areas
                        rect3 = plt.Rectangle(
                            (0, (field_height - 18.3) / 2),
                            5.5,
                            18.3,
                            linewidth=2,
                            edgecolor="white",
                            facecolor="none",
                        )
                        ax.add_patch(rect3)

                        rect4 = plt.Rectangle(
                            (field_width - 5.5, (field_height - 18.3) / 2),
                            5.5,
                            18.3,
                            linewidth=2,
                            edgecolor="white",
                            facecolor="none",
                        )
                        ax.add_patch(rect4)

                    elif sport == "basketball":
                        # Center circle
                        circle = plt.Circle(
                            (field_width / 2, field_height / 2),
                            1.8,
                            edgecolor="white",
                            facecolor="none",
                        )
                        ax.add_patch(circle)

                        # Center line
                        ax.plot(
                            [field_width / 2, field_width / 2],
                            [0, field_height],
                            "white",
                            linestyle="-",
                        )

                        # Three-point arcs
                        arc1 = plt.Circle(
                            (0, field_height / 2),
                            6.75,
                            edgecolor="white",
                            facecolor="none",
                        )
                        ax.add_patch(arc1)

                        arc2 = plt.Circle(
                            (field_width, field_height / 2),
                            6.75,
                            edgecolor="white",
                            facecolor="none",
                        )
                        ax.add_patch(arc2)

                # Generate visualization based on type
                if vis_type == "team":
                    # Get team ID
                    team_id = int(team_var.get().split(" ")[1]) - 1

                    # Generate heatmap data
                    heatmap_data = self.tracker.generate_heatmap_data(team=team_id)

                    if heatmap_data is not None:
                        # Plot heatmap
                        extent = [0, field_width, 0, field_height]
                        im = ax.imshow(
                            heatmap_data,
                            cmap=colormap,
                            origin="lower",
                            extent=extent,
                            aspect="auto",
                            alpha=0.7,
                        )

                        # Add colorbar
                        cbar = fig.colorbar(im, ax=ax)
                        cbar.set_label("Density")

                        ax.set_title(f"Team {team_id+1} Position Heatmap")
                    else:
                        ax.text(
                            0.5,
                            0.5,
                            "No heatmap data available for this team",
                            horizontalalignment="center",
                            verticalalignment="center",
                            transform=ax.transAxes,
                        )

                elif vis_type == "player":
                    # Get player ID
                    player_str = player_var.get()
                    if not player_str:
                        ax.text(
                            0.5,
                            0.5,
                            "Please select a player",
                            horizontalalignment="center",
                            verticalalignment="center",
                            transform=ax.transAxes,
                        )
                    else:
                        player_id = int(player_str.split(" ")[1])

                        # Generate heatmap data
                        heatmap_data = self.tracker.generate_heatmap_data(
                            player_id=player_id
                        )

                        if heatmap_data is not None:
                            # Plot heatmap
                            extent = [0, field_width, 0, field_height]
                            im = ax.imshow(
                                heatmap_data,
                                cmap=colormap,
                                origin="lower",
                                extent=extent,
                                aspect="auto",
                                alpha=0.7,
                            )

                            # Add colorbar
                            cbar = fig.colorbar(im, ax=ax)
                            cbar.set_label("Density")

                            # Get player team for title
                            team = self.tracker.player_teams.get(player_id, -1)
                            team_str = f" (Team {team+1})" if team != -1 else ""

                            ax.set_title(
                                f"Player {player_id}{team_str} Position Heatmap"
                            )
                        else:
                            ax.text(
                                0.5,
                                0.5,
                                "No heatmap data available for this player",
                                horizontalalignment="center",
                                verticalalignment="center",
                                transform=ax.transAxes,
                            )

                elif vis_type == "zone":
                    zone = zone_var.get()
                    if not zone:
                        ax.text(
                            0.5,
                            0.5,
                            "Please select a zone",
                            horizontalalignment="center",
                            verticalalignment="center",
                            transform=ax.transAxes,
                        )
                    else:
                        # Get zone boundaries
                        zone_bounds = CONFIG["sports"][sport]["zones"].get(zone)
                        if zone_bounds:
                            # Highlight the zone
                            (x1, y1), (x2, y2) = zone_bounds
                            rect = plt.Rectangle(
                                (x1, y1),
                                x2 - x1,
                                y2 - y1,
                                linewidth=2,
                                edgecolor="yellow",
                                facecolor="yellow",
                                alpha=0.3,
                            )
                            ax.add_patch(rect)

                            # Count total time spent in zone by each player/team
                            zone_data = {}
                            for player_id, zones in self.tracker.zone_data.items():
                                if zone in zones:
                                    zone_data[player_id] = zones[zone]

                            if zone_data:
                                # Group by team
                                team_data = defaultdict(int)
                                player_labels = []
                                player_values = []
                                player_colors = []

                                for player_id, count in zone_data.items():
                                    team = self.tracker.player_teams.get(player_id, -1)
                                    if team != -1:
                                        team_data[team] += count

                                    player_labels.append(f"Player {player_id}")
                                    player_values.append(count)

                                    # Color based on team
                                    color = (
                                        "red"
                                        if team == 0
                                        else "blue" if team == 1 else "gray"
                                    )
                                    player_colors.append(color)

                                # Create subplots for team and player data
                                ax.remove()  # Remove the main axis

                                ax1 = fig.add_subplot(121)
                                ax2 = fig.add_subplot(122)

                                # Team pie chart
                                if team_data:
                                    labels = [
                                        f"Team {team+1}" for team in team_data.keys()
                                    ]
                                    values = list(team_data.values())
                                    colors = ["red", "blue"]

                                    ax1.pie(
                                        values,
                                        labels=labels,
                                        colors=colors,
                                        autopct="%1.1f%%",
                                        startangle=90,
                                    )
                                    ax1.axis("equal")
                                    ax1.set_title(f"Zone Control: {zone}")
                                else:
                                    ax1.text(
                                        0.5,
                                        0.5,
                                        "No team data",
                                        horizontalalignment="center",
                                        verticalalignment="center",
                                        transform=ax1.transAxes,
                                    )

                                # Player bar chart
                                if player_values:
                                    # Sort by value
                                    sorted_indices = np.argsort(player_values)[
                                        ::-1
                                    ]  # Descending
                                    sorted_labels = [
                                        player_labels[i] for i in sorted_indices
                                    ]
                                    sorted_values = [
                                        player_values[i] for i in sorted_indices
                                    ]
                                    sorted_colors = [
                                        player_colors[i] for i in sorted_indices
                                    ]

                                    # Top 10 players for readability
                                    if len(sorted_labels) > 10:
                                        sorted_labels = sorted_labels[:10]
                                        sorted_values = sorted_values[:10]
                                        sorted_colors = sorted_colors[:10]

                                    y_pos = range(len(sorted_labels))
                                    ax2.barh(y_pos, sorted_values, color=sorted_colors)
                                    ax2.set_yticks(y_pos)
                                    ax2.set_yticklabels(sorted_labels)
                                    ax2.invert_yaxis()  # Labels read top-to-bottom
                                    ax2.set_xlabel("Time in Zone (frames)")
                                    ax2.set_title("Player Zone Time")
                                else:
                                    ax2.text(
                                        0.5,
                                        0.5,
                                        "No player data",
                                        horizontalalignment="center",
                                        verticalalignment="center",
                                        transform=ax2.transAxes,
                                    )
                            else:
                                ax.text(
                                    0.5,
                                    0.5,
                                    "No data available for this zone",
                                    horizontalalignment="center",
                                    verticalalignment="center",
                                    transform=ax.transAxes,
                                )
                        else:
                            ax.text(
                                0.5,
                                0.5,
                                f"Zone boundaries not defined for {zone}",
                                horizontalalignment="center",
                                verticalalignment="center",
                                transform=ax.transAxes,
                            )

                # Common settings
                if (
                    vis_type != "zone" or not zone_data
                ):  # Skip if we've already created custom subplots
                    ax.set_xlabel("X Position (meters)")
                    ax.set_ylabel("Y Position (meters)")
                    ax.set_xlim(0, field_width)
                    ax.set_ylim(0, field_height)

                fig.tight_layout()
                canvas.draw()

            except Exception as e:
                messagebox.showerror("Error", f"Error updating visualization: {str(e)}")

        # Connect callbacks
        type_var.trace("w", update_selection_frame)
        team_var.trace("w", update_visualization)
        player_var.trace("w", update_visualization)
        zone_var.trace("w", update_visualization)
        colormap_var.trace("w", update_visualization)
        field_var.trace("w", update_visualization)
        normalize_var.trace("w", update_visualization)

        # Initial update
        update_selection_frame()

    def show_trajectories(self):
        """Show player trajectories"""
        if not hasattr(self, "tracker") or not self.tracker:
            messagebox.showerror("Error", "No tracking data available.")
            return

        # Create dialog
        dialog = tk.Toplevel(self.root)
        dialog.title("Player Trajectories")
        dialog.geometry("800x700")
        dialog.transient(self.root)

        # Create main frame
        main_frame = ttk.Frame(dialog, padding=10)
        main_frame.pack(fill=tk.BOTH, expand=True)

        # Control panel
        control_frame = ttk.Frame(main_frame)
        control_frame.pack(fill=tk.X, pady=(0, 10))

        # Player selection
        selection_frame = ttk.LabelFrame(control_frame, text="Player Selection")
        selection_frame.pack(side=tk.LEFT, fill=tk.Y, padx=5)

        # Single or multiple player selection
        mode_var = tk.StringVar(value="single")
        ttk.Radiobutton(
            selection_frame, text="Single Player", variable=mode_var, value="single"
        ).pack(anchor=tk.W, padx=10, pady=2)
        ttk.Radiobutton(
            selection_frame,
            text="Multiple Players",
            variable=mode_var,
            value="multiple",
        ).pack(anchor=tk.W, padx=10, pady=2)

        # Player selection
        player_frame = ttk.Frame(control_frame)
        player_frame.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=5)

        # Single player dropdown
        single_frame = ttk.Frame(player_frame)

        ttk.Label(single_frame, text="Select Player:").pack(side=tk.LEFT)

        player_var = tk.StringVar()
        player_combo = ttk.Combobox(
            single_frame,
            textvariable=player_var,
            width=20,
            state="readonly",
            values=[f"Player {pid}" for pid in self.tracker.players.keys()],
        )
        player_combo.pack(side=tk.LEFT, padx=5)
        if self.tracker.players:
            player_combo.current(0)

        # Multiple player selection list
        multi_frame = ttk.Frame(player_frame)

        # Create scrollable checkbutton list
        players_canvas = tk.Canvas(multi_frame, width=200, height=100)
        scrollbar = ttk.Scrollbar(
            multi_frame, orient="vertical", command=players_canvas.yview
        )
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        players_canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        players_canvas.configure(yscrollcommand=scrollbar.set)

        players_content = ttk.Frame(players_canvas)
        players_canvas.create_window((0, 0), window=players_content, anchor="nw")

        # Configure the canvas scrolling
        def configure_players_canvas(event):
            players_canvas.configure(
                scrollregion=players_canvas.bbox("all"), width=event.width
            )

        players_content.bind("<Configure>", configure_players_canvas)

        # Create checkbuttons for each player
        player_vars = {}
        for i, player_id in enumerate(sorted(self.tracker.players.keys())):
            var = tk.BooleanVar(value=False)
            player_vars[player_id] = var

            # Get team for color coding
            team = self.tracker.player_teams.get(player_id, -1)
            team_str = f" (Team {team+1})" if team != -1 else ""

            # Create checkbutton with color based on team
            cb = ttk.Checkbutton(
                players_content, text=f"Player {player_id}{team_str}", variable=var
            )
            cb.pack(anchor=tk.W, padx=5, pady=2)

            # Set foreground color based on team
            if team == 0:
                cb.configure(style="Red.TCheckbutton")
            elif team == 1:
                cb.configure(style="Blue.TCheckbutton")

        # Create styles for colored checkbuttons
        style = ttk.Style()
        style.configure("Red.TCheckbutton", foreground="red")
        style.configure("Blue.TCheckbutton", foreground="blue")

        # Display options
        options_frame = ttk.LabelFrame(control_frame, text="Options")
        options_frame.pack(side=tk.LEFT, fill=tk.Y, padx=5)

        # Show field markings
        field_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(options_frame, text="Show Field", variable=field_var).pack(
            anchor=tk.W, padx=10, pady=5
        )

        # Show start/end points
        points_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(
            options_frame, text="Show Start/End Points", variable=points_var
        ).pack(anchor=tk.W, padx=10, pady=5)

        # Color by time
        time_color_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(
            options_frame, text="Color by Time", variable=time_color_var
        ).pack(anchor=tk.W, padx=10, pady=5)

        # Time range selection
        time_frame = ttk.LabelFrame(options_frame, text="Time Range")
        time_frame.pack(fill=tk.X, padx=5, pady=5)

        # Calculate total time range
        total_frames = self.tracker.current_frame
        frame_rate = self.tracker.frame_rate or 30
        total_seconds = total_frames / frame_rate

        # From time
        from_frame = ttk.Frame(time_frame)
        from_frame.pack(fill=tk.X, padx=5, pady=2)

        ttk.Label(from_frame, text="From:").pack(side=tk.LEFT)

        from_var = tk.DoubleVar(value=0)
        from_scale = ttk.Scale(
            from_frame,
            from_=0,
            to=total_seconds,
            variable=from_var,
            orient=tk.HORIZONTAL,
        )
        from_scale.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=5)

        from_label = ttk.Label(from_frame, text="0:00")
        from_label.pack(side=tk.LEFT)

        # To time
        to_frame = ttk.Frame(time_frame)
        to_frame.pack(fill=tk.X, padx=5, pady=2)

        ttk.Label(to_frame, text="To:").pack(side=tk.LEFT)

        to_var = tk.DoubleVar(value=total_seconds)
        to_scale = ttk.Scale(
            to_frame, from_=0, to=total_seconds, variable=to_var, orient=tk.HORIZONTAL
        )
        to_scale.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=5)

        to_label = ttk.Label(
            to_frame, text=f"{int(total_seconds//60)}:{int(total_seconds%60):02d}"
        )
        to_label.pack(side=tk.LEFT)

        # Function to update time labels
        def update_time_labels(*args):
            from_time = from_var.get()
            to_time = to_var.get()

            # Format as MM:SS
            from_min = int(from_time // 60)
            from_sec = int(from_time % 60)
            from_label.configure(text=f"{from_min}:{from_sec:02d}")

            to_min = int(to_time // 60)
            to_sec = int(to_time % 60)
            to_label.configure(text=f"{to_min}:{to_sec:02d}")

        from_var.trace("w", update_time_labels)
        to_var.trace("w", update_time_labels)

        # Update button
        ttk.Button(
            options_frame,
            text="Update Visualization",
            command=lambda: update_visualization(),
        ).pack(padx=10, pady=10)

        # Visualization area
        vis_frame = ttk.Frame(main_frame)
        vis_frame.pack(fill=tk.BOTH, expand=True, pady=10)

        # Create matplotlib figure
        fig = Figure(figsize=(8, 6), dpi=100)
        canvas = FigureCanvasTkAgg(fig, master=vis_frame)
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        # Toolbar
        from matplotlib.backends.backend_tkagg import NavigationToolbar2Tk

        toolbar = NavigationToolbar2Tk(canvas, vis_frame)
        toolbar.update()

        # Function to update player selection mode
        def update_selection_mode(*args):
            mode = mode_var.get()

            if mode == "single":
                multi_frame.pack_forget()
                single_frame.pack(fill=tk.X, padx=10, pady=10)
            else:  # multiple
                single_frame.pack_forget()
                multi_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        # Function to update visualization
        def update_visualization():
            try:
                mode = mode_var.get()
                show_field = field_var.get()
                show_points = points_var.get()
                color_by_time = time_color_var.get()
                from_time = from_var.get()
                to_time = to_var.get()

                # Convert time to frames
                frame_rate = self.tracker.frame_rate or 30
                from_frame = int(from_time * frame_rate)
                to_frame = int(to_time * frame_rate)

                # Clear figure
                fig.clear()
                ax = fig.add_subplot(111)

                # Get field dimensions for the sport
                sport = self.selected_sport.get()
                field_width, field_height = CONFIG["sports"][sport]["field_dimensions"]

                # Draw field if requested
                if show_field:
                    # Field background
                    rect = plt.Rectangle(
                        (0, 0),
                        field_width,
                        field_height,
                        linewidth=2,
                        edgecolor="green",
                        facecolor="lightgreen",
                        alpha=0.3,
                    )
                    ax.add_patch(rect)

                    # Add field markings based on sport
                    if sport == "football":
                        # Center line
                        ax.plot(
                            [field_width / 2, field_width / 2],
                            [0, field_height],
                            "white",
                            linestyle="-",
                        )

                        # Center circle
                        circle = plt.Circle(
                            (field_width / 2, field_height / 2),
                            9.15,
                            edgecolor="white",
                            facecolor="none",
                        )
                        ax.add_patch(circle)

                        # Penalty areas
                        rect1 = plt.Rectangle(
                            (0, (field_height - 40.3) / 2),
                            16.5,
                            40.3,
                            linewidth=2,
                            edgecolor="white",
                            facecolor="none",
                        )
                        ax.add_patch(rect1)

                        rect2 = plt.Rectangle(
                            (field_width - 16.5, (field_height - 40.3) / 2),
                            16.5,
                            40.3,
                            linewidth=2,
                            edgecolor="white",
                            facecolor="none",
                        )
                        ax.add_patch(rect2)

                # Get selected players
                selected_players = []

                if mode == "single":
                    # Get single player ID
                    player_str = player_var.get()
                    if player_str:
                        player_id = int(player_str.split(" ")[1])
                        selected_players.append(player_id)
                else:  # multiple
                    # Get all selected players
                    for player_id, var in player_vars.items():
                        if var.get():
                            selected_players.append(player_id)

                if not selected_players:
                    ax.text(
                        0.5,
                        0.5,
                        "No players selected",
                        horizontalalignment="center",
                        verticalalignment="center",
                        transform=ax.transAxes,
                    )
                else:
                    # Plot trajectories for each selected player
                    for player_id in selected_players:
                        if player_id not in self.tracker.players:
                            continue

                        positions = self.tracker.players[player_id]

                        # Skip if no positions
                        # Skip if no positions
                        if not positions:
                            continue

                        # Get position indices within the specified time range
                        # This is a simplification - in a full implementation we would need proper frame mapping
                        frames_to_include = min(len(positions), to_frame) - from_frame
                        if frames_to_include <= 0:
                            continue

                        filtered_positions = positions[
                            max(0, from_frame) : min(len(positions), to_frame)
                        ]

                        # Skip if too few positions after filtering
                        if len(filtered_positions) < 2:
                            continue

                        # Extract x and y coordinates
                        x_coords, y_coords = zip(*filtered_positions)

                        # Get player team for color
                        team = self.tracker.player_teams.get(player_id, -1)
                        base_color = (
                            "red" if team == 0 else "blue" if team == 1 else "gray"
                        )

                        # Plot the trajectory
                        if color_by_time:
                            # Create colormap for time progression
                            norm = Normalize(vmin=0, vmax=len(filtered_positions) - 1)
                            colors = [
                                plt.cm.viridis(norm(i))
                                for i in range(len(filtered_positions))
                            ]

                            # Plot segments with color gradient
                            for i in range(len(filtered_positions) - 1):
                                ax.plot(
                                    [x_coords[i], x_coords[i + 1]],
                                    [y_coords[i], y_coords[i + 1]],
                                    color=colors[i],
                                    linewidth=2,
                                )
                        else:
                            # Plot whole trajectory with team color
                            ax.plot(
                                x_coords,
                                y_coords,
                                color=base_color,
                                linewidth=2,
                                label=f"Player {player_id}",
                            )

                        # Show start/end points if requested
                        if show_points:
                            ax.plot(
                                x_coords[0],
                                y_coords[0],
                                "go",
                                markersize=8,
                                markeredgecolor="white",
                            )
                            ax.plot(
                                x_coords[-1],
                                y_coords[-1],
                                "ro",
                                markersize=8,
                                markeredgecolor="white",
                            )

                            # Add player ID label near end point
                            ax.annotate(
                                f"P{player_id}",
                                (x_coords[-1], y_coords[-1]),
                                xytext=(5, 5),
                                textcoords="offset points",
                            )

                # Set plot limits and labels
                ax.set_xlim(0, field_width)
                ax.set_ylim(0, field_height)
                ax.set_xlabel("X Position (meters)")
                ax.set_ylabel("Y Position (meters)")

                # Add legend if multiple players and not using time coloring
                if len(selected_players) > 1 and not color_by_time:
                    ax.legend(loc="upper right")

                # Set title
                if mode == "single" and selected_players:
                    player_id = selected_players[0]
                    team = self.tracker.player_teams.get(player_id, -1)
                    team_str = f" (Team {team+1})" if team != -1 else ""
                    ax.set_title(f"Player {player_id}{team_str} Trajectory")
                else:
                    ax.set_title(
                        f"Player Trajectories ({len(selected_players)} players)"
                    )

                fig.tight_layout()
                canvas.draw()

            except Exception as e:
                messagebox.showerror("Error", f"Error updating visualization: {str(e)}")

        # Connect callbacks
        mode_var.trace("w", update_selection_mode)

        # Initial setup
        update_selection_mode()
        update_time_labels()
        update_visualization()

    def show_speed_profiles(self):
        """Show player speed profiles"""
        if not hasattr(self, "tracker") or not self.tracker:
            messagebox.showerror("Error", "No tracking data available.")
            return

        # Create dialog
        dialog = tk.Toplevel(self.root)
        dialog.title("Speed Profile Analysis")
        dialog.geometry("800x700")
        dialog.transient(self.root)

        # Create main frame
        main_frame = ttk.Frame(dialog, padding=10)
        main_frame.pack(fill=tk.BOTH, expand=True)

        # Control panel
        control_frame = ttk.Frame(main_frame)
        control_frame.pack(fill=tk.X, pady=(0, 10))

        # Player selection
        selection_frame = ttk.LabelFrame(control_frame, text="Player Selection")
        selection_frame.pack(side=tk.LEFT, fill=tk.Y, padx=5)

        # Single or multiple player selection
        mode_var = tk.StringVar(value="single")
        ttk.Radiobutton(
            selection_frame, text="Single Player", variable=mode_var, value="single"
        ).pack(anchor=tk.W, padx=10, pady=2)
        ttk.Radiobutton(
            selection_frame,
            text="Multiple Players",
            variable=mode_var,
            value="multiple",
        ).pack(anchor=tk.W, padx=10, pady=2)
        ttk.Radiobutton(
            selection_frame, text="Team Comparison", variable=mode_var, value="team"
        ).pack(anchor=tk.W, padx=10, pady=2)

        # Player selection
        player_frame = ttk.Frame(control_frame)
        player_frame.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=5)

        # Single player dropdown
        single_frame = ttk.Frame(player_frame)

        ttk.Label(single_frame, text="Select Player:").pack(side=tk.LEFT)

        player_var = tk.StringVar()
        player_combo = ttk.Combobox(
            single_frame,
            textvariable=player_var,
            width=20,
            state="readonly",
            values=[f"Player {pid}" for pid in self.tracker.players.keys()],
        )
        player_combo.pack(side=tk.LEFT, padx=5)
        if self.tracker.players:
            player_combo.current(0)

        # Multiple player selection list
        multi_frame = ttk.Frame(player_frame)

        # Create scrollable checkbutton list
        players_canvas = tk.Canvas(multi_frame, width=200, height=100)
        scrollbar = ttk.Scrollbar(
            multi_frame, orient="vertical", command=players_canvas.yview
        )
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        players_canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        players_canvas.configure(yscrollcommand=scrollbar.set)

        players_content = ttk.Frame(players_canvas)
        players_canvas.create_window((0, 0), window=players_content, anchor="nw")

        # Configure the canvas scrolling
        def configure_players_canvas(event):
            players_canvas.configure(
                scrollregion=players_canvas.bbox("all"), width=event.width
            )

        players_content.bind("<Configure>", configure_players_canvas)

        # Create checkbuttons for each player
        player_vars = {}
        for i, player_id in enumerate(sorted(self.tracker.players.keys())):
            var = tk.BooleanVar(value=False)
            player_vars[player_id] = var

            # Get team for color coding
            team = self.tracker.player_teams.get(player_id, -1)
            team_str = f" (Team {team+1})" if team != -1 else ""

            cb = ttk.Checkbutton(
                players_content, text=f"Player {player_id}{team_str}", variable=var
            )
            cb.pack(anchor=tk.W, padx=5, pady=2)

            # Set foreground color based on team
            if team == 0:
                cb.configure(style="Red.TCheckbutton")
            elif team == 1:
                cb.configure(style="Blue.TCheckbutton")

        # Team comparison frame
        team_frame = ttk.Frame(player_frame)

        ttk.Label(team_frame, text="Select teams to compare:").pack(
            anchor=tk.W, padx=5, pady=5
        )

        team1_var = tk.BooleanVar(value=True)
        team2_var = tk.BooleanVar(value=True)

        ttk.Checkbutton(
            team_frame, text="Team 1", variable=team1_var, style="Red.TCheckbutton"
        ).pack(anchor=tk.W, padx=5, pady=2)
        ttk.Checkbutton(
            team_frame, text="Team 2", variable=team2_var, style="Blue.TCheckbutton"
        ).pack(anchor=tk.W, padx=5, pady=2)

        # Display options
        options_frame = ttk.LabelFrame(control_frame, text="Options")
        options_frame.pack(side=tk.LEFT, fill=tk.Y, padx=5)

        # Show threshold lines
        threshold_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(
            options_frame, text="Show Speed Thresholds", variable=threshold_var
        ).pack(anchor=tk.W, padx=10, pady=5)

        # Show average line
        avg_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(
            options_frame, text="Show Average Speed", variable=avg_var
        ).pack(anchor=tk.W, padx=10, pady=5)

        # Show max speed
        max_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(
            options_frame, text="Highlight Max Speed", variable=max_var
        ).pack(anchor=tk.W, padx=10, pady=5)

        # Display type
        display_frame = ttk.Frame(options_frame)
        display_frame.pack(fill=tk.X, padx=5, pady=5)

        ttk.Label(display_frame, text="Display:").pack(anchor=tk.W, padx=5, pady=2)

        display_var = tk.StringVar(value="line")
        ttk.Radiobutton(
            display_frame, text="Line Chart", variable=display_var, value="line"
        ).pack(anchor=tk.W, padx=15, pady=2)
        ttk.Radiobutton(
            display_frame, text="Bar Chart", variable=display_var, value="bar"
        ).pack(anchor=tk.W, padx=15, pady=2)
        ttk.Radiobutton(
            display_frame, text="Distribution", variable=display_var, value="dist"
        ).pack(anchor=tk.W, padx=15, pady=2)

        # Update button
        ttk.Button(
            options_frame,
            text="Update Visualization",
            command=lambda: update_visualization(),
        ).pack(padx=10, pady=10)

        # Visualization area
        vis_frame = ttk.Frame(main_frame)
        vis_frame.pack(fill=tk.BOTH, expand=True, pady=10)

        # Create matplotlib figure
        fig = Figure(figsize=(8, 6), dpi=100)
        canvas = FigureCanvasTkAgg(fig, master=vis_frame)
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        # Toolbar
        from matplotlib.backends.backend_tkagg import NavigationToolbar2Tk

        toolbar = NavigationToolbar2Tk(canvas, vis_frame)
        toolbar.update()

        # Function to update player selection mode
        def update_selection_mode(*args):
            mode = mode_var.get()

            # Hide all frames first
            single_frame.pack_forget()
            multi_frame.pack_forget()
            team_frame.pack_forget()

            if mode == "single":
                single_frame.pack(fill=tk.X, padx=10, pady=10)
            elif mode == "multiple":
                multi_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
            else:  # team
                team_frame.pack(fill=tk.X, padx=10, pady=10)

        # Function to update visualization
        def update_visualization():
            try:
                mode = mode_var.get()
                show_thresholds = threshold_var.get()
                show_avg = avg_var.get()
                show_max = max_var.get()
                display_type = display_var.get()

                # Clear figure
                fig.clear()

                # Get selected players/teams
                selected_players = []

                if mode == "single":
                    # Get single player ID
                    player_str = player_var.get()
                    if player_str:
                        player_id = int(player_str.split(" ")[1])
                        selected_players.append(player_id)
                elif mode == "multiple":
                    # Get all selected players
                    for player_id, var in player_vars.items():
                        if var.get():
                            selected_players.append(player_id)
                else:  # team
                    # Get all players from selected teams
                    teams_to_include = []
                    if team1_var.get():
                        teams_to_include.append(0)
                    if team2_var.get():
                        teams_to_include.append(1)

                    for player_id, team in self.tracker.player_teams.items():
                        if team in teams_to_include:
                            selected_players.append(player_id)

                if not selected_players:
                    ax = fig.add_subplot(111)
                    ax.text(
                        0.5,
                        0.5,
                        "No players selected",
                        horizontalalignment="center",
                        verticalalignment="center",
                        transform=ax.transAxes,
                    )
                else:
                    # Different visualizations based on display type
                    if display_type == "line":
                        # Line chart of speed over time
                        ax = fig.add_subplot(111)

                        # Plot for each player
                        for player_id in selected_players:
                            if player_id not in self.tracker.player_speeds:
                                continue

                            speeds = self.tracker.player_speeds.get(player_id, [])

                            # Skip if no speeds
                            if not speeds:
                                continue

                            # Create time axis
                            frame_rate = self.tracker.frame_rate or 30
                            time_points = [i / frame_rate for i in range(len(speeds))]

                            # Get player team for color
                            team = self.tracker.player_teams.get(player_id, -1)
                            color = (
                                "red" if team == 0 else "blue" if team == 1 else "gray"
                            )

                            # Plot speed line
                            (line,) = ax.plot(
                                time_points,
                                speeds,
                                color=color,
                                alpha=0.7,
                                label=f"Player {player_id}",
                            )

                            # Show max speed point if requested
                            if show_max and speeds:
                                max_speed = max(speeds)
                                max_idx = speeds.index(max_speed)
                                max_time = time_points[max_idx]

                                ax.plot(
                                    max_time,
                                    max_speed,
                                    "o",
                                    color=color,
                                    markersize=6,
                                    markeredgecolor="white",
                                )

                                if (
                                    len(selected_players) <= 3
                                ):  # Only annotate if few players
                                    ax.annotate(
                                        f"{max_speed:.2f} m/s",
                                        (max_time, max_speed),
                                        xytext=(5, 5),
                                        textcoords="offset points",
                                    )

                            # Show average line if requested
                            if show_avg and speeds:
                                avg_speed = sum(speeds) / len(speeds)
                                ax.axhline(
                                    y=avg_speed, color=color, linestyle="--", alpha=0.5
                                )

                        # Show speed thresholds if requested
                        if show_thresholds:
                            thresholds = [
                                (2.0, "Walking"),
                                (4.0, "Jogging"),
                                (7.0, "Running"),
                                (9.0, "Sprinting"),
                            ]
                            colors = ["green", "orange", "red", "purple"]

                            for i, (threshold, label) in enumerate(thresholds):
                                ax.axhline(
                                    y=threshold,
                                    color=colors[i],
                                    linestyle=":",
                                    alpha=0.5,
                                    label=f"{label} ({threshold} m/s)",
                                )

                        ax.set_xlabel("Time (seconds)")
                        ax.set_ylabel("Speed (m/s)")
                        ax.grid(True, alpha=0.3)

                        # Add legend if multiple players or thresholds
                        if len(selected_players) > 1 or show_thresholds:
                            ax.legend()

                        # Set title based on mode
                        if mode == "single" and selected_players:
                            player_id = selected_players[0]
                            team = self.tracker.player_teams.get(player_id, -1)
                            team_str = f" (Team {team+1})" if team != -1 else ""
                            ax.set_title(f"Player {player_id}{team_str} Speed Profile")
                        elif mode == "team":
                            selected_teams = []
                            if team1_var.get():
                                selected_teams.append("Team 1")
                            if team2_var.get():
                                selected_teams.append("Team 2")
                            teams_str = " vs ".join(selected_teams)
                            ax.set_title(f"Team Speed Comparison: {teams_str}")
                        else:
                            ax.set_title(
                                f"Speed Profiles ({len(selected_players)} players)"
                            )

                    elif display_type == "bar":
                        # Bar chart of average and max speeds
                        ax = fig.add_subplot(111)

                        player_labels = []
                        avg_speeds = []
                        max_speeds = []
                        colors = []

                        for player_id in selected_players:
                            speeds = self.tracker.player_speeds.get(player_id, [])

                            # Skip if no speeds
                            if not speeds:
                                continue

                            player_labels.append(f"P{player_id}")
                            avg_speeds.append(sum(speeds) / len(speeds))
                            max_speeds.append(max(speeds))

                            # Get color based on team
                            team = self.tracker.player_teams.get(player_id, -1)
                            color = (
                                "red" if team == 0 else "blue" if team == 1 else "gray"
                            )
                            colors.append(color)

                        if not player_labels:
                            ax.text(
                                0.5,
                                0.5,
                                "No speed data available",
                                horizontalalignment="center",
                                verticalalignment="center",
                                transform=ax.transAxes,
                            )
                        else:
                            # Position for bars
                            x = range(len(player_labels))
                            width = 0.35

                            # Create bars
                            avg_bars = ax.bar(
                                [i - width / 2 for i in x],
                                avg_speeds,
                                width,
                                label="Avg Speed",
                                alpha=0.7,
                            )
                            max_bars = ax.bar(
                                [i + width / 2 for i in x],
                                max_speeds,
                                width,
                                label="Max Speed",
                                alpha=0.7,
                            )

                            # Color bars based on team
                            for i, color in enumerate(colors):
                                avg_bars[i].set_color(color)
                                max_bars[i].set_color(color)
                                # Add darker edge
                                avg_bars[i].set_edgecolor("black")
                                max_bars[i].set_edgecolor("black")

                            # Add value labels on bars
                            for bar in avg_bars:
                                height = bar.get_height()
                                ax.text(
                                    bar.get_x() + bar.get_width() / 2.0,
                                    height + 0.1,
                                    f"{height:.1f}",
                                    ha="center",
                                    va="bottom",
                                    fontsize=8,
                                )

                            for bar in max_bars:
                                height = bar.get_height()
                                ax.text(
                                    bar.get_x() + bar.get_width() / 2.0,
                                    height + 0.1,
                                    f"{height:.1f}",
                                    ha="center",
                                    va="bottom",
                                    fontsize=8,
                                )

                            # Show thresholds if requested
                            if show_thresholds:
                                thresholds = [
                                    (2.0, "Walking"),
                                    (4.0, "Jogging"),
                                    (7.0, "Running"),
                                    (9.0, "Sprinting"),
                                ]
                                colors = ["green", "orange", "red", "purple"]

                                for i, (threshold, label) in enumerate(thresholds):
                                    ax.axhline(
                                        y=threshold,
                                        color=colors[i],
                                        linestyle=":",
                                        alpha=0.5,
                                        label=f"{label}",
                                    )

                            ax.set_xlabel("Player")
                            ax.set_ylabel("Speed (m/s)")
                            ax.set_xticks(x)
                            ax.set_xticklabels(player_labels)
                            ax.legend()

                            # Set title based on mode
                            if mode == "team":
                                selected_teams = []
                                if team1_var.get():
                                    selected_teams.append("Team 1")
                                if team2_var.get():
                                    selected_teams.append("Team 2")
                                teams_str = " vs ".join(selected_teams)
                                ax.set_title(f"Team Speed Comparison: {teams_str}")
                            else:
                                ax.set_title("Player Speed Comparison")

                    elif display_type == "dist":
                        # Speed distribution (histogram)
                        ax = fig.add_subplot(111)

                        # Group speeds by team if in team mode
                        if mode == "team":
                            team_speeds = {0: [], 1: []}

                            for player_id in selected_players:
                                speeds = self.tracker.player_speeds.get(player_id, [])
                                team = self.tracker.player_teams.get(player_id, -1)

                                if team in [0, 1] and speeds:
                                    team_speeds[team].extend(speeds)

                            # Plot histogram for each team
                            for team, speeds in team_speeds.items():
                                if not speeds:
                                    continue

                                color = "red" if team == 0 else "blue"
                                label = f"Team {team+1}"

                                ax.hist(
                                    speeds, bins=20, color=color, alpha=0.5, label=label
                                )
                        else:
                            # Plot histogram for each player
                            for player_id in selected_players:
                                speeds = self.tracker.player_speeds.get(player_id, [])

                                if not speeds:
                                    continue

                                team = self.tracker.player_teams.get(player_id, -1)
                                color = (
                                    "red"
                                    if team == 0
                                    else "blue" if team == 1 else "gray"
                                )

                                ax.hist(
                                    speeds,
                                    bins=15,
                                    color=color,
                                    alpha=0.5,
                                    label=f"Player {player_id}",
                                )

                        # Show thresholds if requested
                        if show_thresholds:
                            thresholds = [
                                (2.0, "Walking"),
                                (4.0, "Jogging"),
                                (7.0, "Running"),
                                (9.0, "Sprinting"),
                            ]
                            colors = ["green", "orange", "red", "purple"]

                            for i, (threshold, label) in enumerate(thresholds):
                                ax.axvline(
                                    x=threshold,
                                    color=colors[i],
                                    linestyle=":",
                                    alpha=0.7,
                                    label=f"{label}",
                                )

                        ax.set_xlabel("Speed (m/s)")
                        ax.set_ylabel("Frequency")
                        ax.legend()

                        # Set title
                        if mode == "team":
                            ax.set_title("Team Speed Distribution")
                        elif mode == "single" and selected_players:
                            player_id = selected_players[0]
                            ax.set_title(f"Player {player_id} Speed Distribution")
                        else:
                            ax.set_title("Player Speed Distribution")

                fig.tight_layout()
                canvas.draw()

            except Exception as e:
                messagebox.showerror("Error", f"Error updating visualization: {str(e)}")

        # Connect callbacks
        mode_var.trace("w", update_selection_mode)

        # Initial setup
        update_selection_mode()
        update_visualization()

    def show_interaction_network(self):
        """Show player interaction network"""
        if not hasattr(self, "tracker") or not self.tracker:
            messagebox.showerror("Error", "No tracking data available.")
            return

        # Check if we have team data
        if not self.tracker.player_teams:
            messagebox.showerror(
                "Error",
                "No team data available. Enable team detection during tracking.",
            )
            return

        # Analyze interactions
        try:
            interaction_data = self.tracker.analyze_interactions(distance_threshold=5)
            if not interaction_data or not interaction_data.get("player_interactions"):
                messagebox.showerror("Error", "No interaction data available.")
                return

            player_interactions = interaction_data["player_interactions"]
            team_interactions = interaction_data.get("team_interactions", {})
        except Exception as e:
            messagebox.showerror("Error", f"Error analyzing interactions: {str(e)}")
            return

        # Create dialog
        dialog = tk.Toplevel(self.root)
        dialog.title("Player Interaction Network")
        dialog.geometry("900x700")
        dialog.transient(self.root)

        # Create main frame
        main_frame = ttk.Frame(dialog, padding=10)
        main_frame.pack(fill=tk.BOTH, expand=True)

        # Create notebook for different views
        notebook = ttk.Notebook(main_frame)
        notebook.pack(fill=tk.BOTH, expand=True)

        # Network tab
        network_tab = ttk.Frame(notebook)
        notebook.add(network_tab, text="Interaction Network")

        # Matrix tab
        matrix_tab = ttk.Frame(notebook)
        notebook.add(matrix_tab, text="Interaction Matrix")

        # Team tab
        team_tab = ttk.Frame(notebook)
        notebook.add(team_tab, text="Team Interactions")

        # Stats tab
        stats_tab = ttk.Frame(notebook)
        notebook.add(stats_tab, text="Interaction Stats")

        # Control panel for network tab
        network_control_frame = ttk.Frame(network_tab)
        network_control_frame.pack(fill=tk.X, pady=(0, 10))

        # Options
        options_frame = ttk.LabelFrame(network_control_frame, text="Display Options")
        options_frame.pack(side=tk.LEFT, fill=tk.Y, padx=5)

        # Show player labels
        labels_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(
            options_frame, text="Show Player Labels", variable=labels_var
        ).pack(anchor=tk.W, padx=10, pady=5)

        # Show team colors
        team_color_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(
            options_frame, text="Show Team Colors", variable=team_color_var
        ).pack(anchor=tk.W, padx=10, pady=5)

        # Show field background
        field_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(options_frame, text="Show Field", variable=field_var).pack(
            anchor=tk.W, padx=10, pady=5
        )

        # Interaction threshold
        threshold_frame = ttk.Frame(options_frame)
        threshold_frame.pack(fill=tk.X, padx=10, pady=5)

        ttk.Label(threshold_frame, text="Min Interactions:").pack(side=tk.LEFT)

        threshold_var = tk.IntVar(value=5)
        threshold_spinbox = ttk.Spinbox(
            threshold_frame, from_=1, to=100, textvariable=threshold_var, width=5
        )
        threshold_spinbox.pack(side=tk.LEFT, padx=5)

        # Layout options
        layout_frame = ttk.Frame(options_frame)
        layout_frame.pack(fill=tk.X, padx=10, pady=5)

        ttk.Label(layout_frame, text="Layout:").pack(side=tk.LEFT)

        layout_var = tk.StringVar(value="position")
        layout_combo = ttk.Combobox(
            layout_frame,
            textvariable=layout_var,
            values=["position", "spring", "circular"],
            state="readonly",
            width=10,
        )
        layout_combo.pack(side=tk.LEFT, padx=5)

        # Update button
        ttk.Button(
            options_frame, text="Update Network", command=lambda: update_network()
        ).pack(padx=10, pady=10)

        # Filtering options
        filter_frame = ttk.LabelFrame(network_control_frame, text="Filter")
        filter_frame.pack(side=tk.LEFT, fill=tk.Y, padx=5)

        # Filter by team
        team_frame = ttk.Frame(filter_frame)
        team_frame.pack(fill=tk.X, padx=10, pady=5)

        ttk.Label(team_frame, text="Teams:").pack(anchor=tk.W)

        team1_var = tk.BooleanVar(value=True)
        team2_var = tk.BooleanVar(value=True)
        cross_var = tk.BooleanVar(value=True)

        ttk.Checkbutton(
            team_frame,
            text="Team 1 Internal",
            variable=team1_var,
            style="Red.TCheckbutton",
        ).pack(anchor=tk.W, padx=20, pady=2)
        ttk.Checkbutton(
            team_frame,
            text="Team 2 Internal",
            variable=team2_var,
            style="Blue.TCheckbutton",
        ).pack(anchor=tk.W, padx=20, pady=2)
        ttk.Checkbutton(team_frame, text="Cross-team", variable=cross_var).pack(
            anchor=tk.W, padx=20, pady=2
        )

        # Filter by specific player
        player_frame = ttk.Frame(filter_frame)
        player_frame.pack(fill=tk.X, padx=10, pady=5)

        ttk.Label(player_frame, text="Focus Player:").pack(anchor=tk.W)

        focus_var = tk.StringVar(value="All")
        focus_combo = ttk.Combobox(
            player_frame,
            textvariable=focus_var,
            values=["All"] + [f"Player {pid}" for pid in self.tracker.players.keys()],
            state="readonly",
            width=15,
        )
        focus_combo.pack(anchor=tk.W, padx=20, pady=2)

        # Network visualization area
        network_vis_frame = ttk.Frame(network_tab)
        network_vis_frame.pack(fill=tk.BOTH, expand=True, pady=10)

        # Create matplotlib figure for network
        network_fig = Figure(figsize=(8, 6), dpi=100)
        network_canvas = FigureCanvasTkAgg(network_fig, master=network_vis_frame)
        network_canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        # Toolbar
        from matplotlib.backends.backend_tkagg import NavigationToolbar2Tk

        network_toolbar = NavigationToolbar2Tk(network_canvas, network_vis_frame)
        network_toolbar.update()

        # Matrix visualization area
        matrix_frame = ttk.Frame(matrix_tab)
        matrix_frame.pack(fill=tk.BOTH, expand=True, pady=10)

        # Create matplotlib figure for matrix
        matrix_fig = Figure(figsize=(8, 6), dpi=100)
        matrix_canvas = FigureCanvasTkAgg(matrix_fig, master=matrix_frame)
        matrix_canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        # Team interaction visualization
        team_frame = ttk.Frame(team_tab)
        team_frame.pack(fill=tk.BOTH, expand=True, pady=10)

        # Create matplotlib figure for team interactions
        team_fig = Figure(figsize=(8, 6), dpi=100)
        team_canvas = FigureCanvasTkAgg(team_fig, master=team_frame)
        team_canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        # Stats visualization
        stats_frame = ttk.Frame(stats_tab)
        stats_frame.pack(fill=tk.BOTH, expand=True, pady=10)

        # Function to update network visualization
        def update_network():
            try:
                show_labels = labels_var.get()
                show_team_colors = team_color_var.get()
                show_field = field_var.get()
                min_interactions = threshold_var.get()
                layout_type = layout_var.get()

                # Get team filters
                show_team1 = team1_var.get()
                show_team2 = team2_var.get()
                show_cross = cross_var.get()

                # Get focus player
                focus_player = focus_var.get()
                focus_id = (
                    None if focus_player == "All" else int(focus_player.split(" ")[1])
                )

                # Clear figure
                network_fig.clear()
                ax = network_fig.add_subplot(111)

                # Get field dimensions for the sport
                sport = self.selected_sport.get()
                field_width, field_height = CONFIG["sports"][sport]["field_dimensions"]

                # Calculate average position for each player
                player_positions = {}
                for player_id, positions in self.tracker.players.items():
                    if positions:
                        avg_pos = np.mean(positions, axis=0)
                        player_positions[player_id] = avg_pos

                # Filter interactions based on options
                filtered_interactions = {}

                for (p1, p2), count in player_interactions.items():
                    # Skip if below threshold
                    if count < min_interactions:
                        continue

                    # Skip if players don't have position data
                    if p1 not in player_positions or p2 not in player_positions:
                        continue

                    # Skip if focus player is set and neither player matches
                    if focus_id is not None and p1 != focus_id and p2 != focus_id:
                        continue

                    # Check team filters
                    team1 = self.tracker.player_teams.get(p1, -1)
                    team2 = self.tracker.player_teams.get(p2, -1)

                    if team1 == 0 and team2 == 0 and not show_team1:
                        continue
                    if team1 == 1 and team2 == 1 and not show_team2:
                        continue
                    if team1 != team2 and not show_cross:
                        continue

                    filtered_interactions[(p1, p2)] = count

                # Calculate node positions based on layout
                if layout_type == "position":
                    # Use average positions on field
                    node_positions = player_positions
                else:
                    # Create NetworkX graph for layout
                    G = nx.Graph()
                    for player_id in player_positions:
                        G.add_node(player_id)

                    for (p1, p2), count in filtered_interactions.items():
                        G.add_edge(p1, p2, weight=count)

                    if layout_type == "spring":
                        # Use force-directed layout
                        pos = nx.spring_layout(G, weight="weight", seed=42)
                        # Scale to field dimensions
                        node_positions = {
                            n: (
                                p[0] * field_width * 0.8 + field_width * 0.1,
                                p[1] * field_height * 0.8 + field_height * 0.1,
                            )
                            for n, p in pos.items()
                        }
                    elif layout_type == "circular":
                        # Use circular layout
                        pos = nx.circular_layout(G)
                        # Scale to field dimensions
                        node_positions = {
                            n: (
                                p[0] * field_width * 0.4 + field_width * 0.5,
                                p[1] * field_height * 0.4 + field_height * 0.5,
                            )
                            for n, p in pos.items()
                        }

                # Draw field if requested
                if show_field:
                    # Field background
                    rect = plt.Rectangle(
                        (0, 0),
                        field_width,
                        field_height,
                        linewidth=2,
                        edgecolor="green",
                        facecolor="lightgreen",
                        alpha=0.2,
                    )
                    ax.add_patch(rect)

                    # Add field markings based on sport
                    if sport == "football":
                        # Center line
                        ax.plot(
                            [field_width / 2, field_width / 2],
                            [0, field_height],
                            "white",
                            linestyle="-",
                        )

                        # Center circle
                        circle = plt.Circle(
                            (field_width / 2, field_height / 2),
                            9.15,
                            edgecolor="white",
                            facecolor="none",
                        )
                        ax.add_patch(circle)

                # Draw edges (interactions)
                max_count = (
                    max(filtered_interactions.values()) if filtered_interactions else 1
                )

                for (p1, p2), count in filtered_interactions.items():
                    if p1 in node_positions and p2 in node_positions:
                        pos1 = node_positions[p1]
                        pos2 = node_positions[p2]

                        # Width based on interaction count
                        width = 1 + (count / max_count) * 5

                        # Color based on teams
                        team1 = self.tracker.player_teams.get(p1, -1)
                        team2 = self.tracker.player_teams.get(p2, -1)

                        if team1 == team2:
                            if team1 == 0:
                                color = "red"
                            elif team1 == 1:
                                color = "blue"
                            else:
                                color = "gray"
                        else:
                            color = "purple"

                        # Draw line with alpha based on interaction strength
                        alpha = 0.3 + (count / max_count) * 0.7
                        ax.plot(
                            [pos1[0], pos2[0]],
                            [pos1[1], pos2[1]],
                            color=color,
                            linewidth=width,
                            alpha=alpha,
                        )

                # Draw nodes (players)
                for player_id, pos in node_positions.items():
                    # Skip players with no interactions in filtered set
                    has_interaction = any(
                        player_id in pair for pair in filtered_interactions.keys()
                    )
                    if not has_interaction and focus_id != player_id:
                        continue

                    # Size based on number of interactions
                    interactions = sum(
                        count
                        for (p1, p2), count in filtered_interactions.items()
                        if p1 == player_id or p2 == player_id
                    )
                    size = 100 + interactions * 10 / max_count if max_count > 0 else 100

                    # Color based on team
                    team = self.tracker.player_teams.get(player_id, -1)
                    if show_team_colors:
                        color = "red" if team == 0 else "blue" if team == 1 else "gray"
                    else:
                        color = "gray"

                    # Highlight focus player
                    if focus_id == player_id:
                        edgecolor = "yellow"
                        linewidth = 2
                        size *= 1.5
                    else:
                        edgecolor = "white"
                        linewidth = 1

                    # Draw player node
                    ax.scatter(
                        pos[0],
                        pos[1],
                        s=size,
                        color=color,
                        edgecolor=edgecolor,
                        linewidth=linewidth,
                        zorder=10,
                    )

                    # Add player label
                    if show_labels:
                        ax.text(
                            pos[0],
                            pos[1] + 2,
                            f"{player_id}",
                            ha="center",
                            va="center",
                            color="black",
                            fontweight="bold",
                            fontsize=9,
                            bbox=dict(
                                facecolor="white", alpha=0.7, edgecolor="none", pad=1
                            ),
                            zorder=11,
                        )

                # Set plot limits and labels
                if layout_type == "position":
                    ax.set_xlim(0, field_width)
                    ax.set_ylim(0, field_height)
                else:
                    margin = 0.1
                    ax.set_xlim(-margin * field_width, field_width * (1 + margin))
                    ax.set_ylim(-margin * field_height, field_height * (1 + margin))

                ax.set_xlabel("X Position (meters)")
                ax.set_ylabel("Y Position (meters)")

                # Focus player in title if set
                if focus_id is not None:
                    team = self.tracker.player_teams.get(focus_id, -1)
                    team_str = f" (Team {team+1})" if team != -1 else ""
                    ax.set_title(f"Player {focus_id}{team_str} Interaction Network")
                else:
                    ax.set_title("Player Interaction Network")

                network_fig.tight_layout()
                network_canvas.draw()

                # Also update the matrix visualization
                update_matrix()
                update_team_interactions()
                update_stats()

            except Exception as e:
                messagebox.showerror(
                    "Error", f"Error updating network visualization: {str(e)}"
                )

        # Function to update matrix visualization
        def update_matrix():
            try:
                # Clear figure
                matrix_fig.clear()
                ax = matrix_fig.add_subplot(111)

                # Get all players with team info
                players = []
                for player_id in sorted(self.tracker.players.keys()):
                    team = self.tracker.player_teams.get(player_id, -1)
                    if team != -1:  # Only include players with team assignment
                        players.append((player_id, team))

                if not players:
                    ax.text(
                        0.5,
                        0.5,
                        "No player data available",
                        horizontalalignment="center",
                        verticalalignment="center",
                        transform=ax.transAxes,
                    )
                    matrix_canvas.draw()
                    return

                # Sort players by team, then ID
                players.sort(key=lambda x: (x[1], x[0]))

                # Extract player IDs in order
                player_ids = [p[0] for p in players]
                player_teams = [p[1] for p in players]

                # Create interaction matrix
                n = len(player_ids)
                matrix = np.zeros((n, n))

                # Fill in interaction counts
                for i, p1 in enumerate(player_ids):
                    for j, p2 in enumerate(player_ids):
                        if i == j:
                            matrix[i, j] = 0  # No self-interactions
                        else:
                            # Find interaction count (in either direction)
                            pair = tuple(sorted([p1, p2]))
                            matrix[i, j] = player_interactions.get(pair, 0)

                # Create heatmap
                im = ax.imshow(matrix, cmap="viridis")

                # Add colorbar
                cbar = matrix_fig.colorbar(im, ax=ax)
                cbar.set_label("Interaction Count")

                # Add player labels
                ax.set_xticks(range(n))
                ax.set_yticks(range(n))

                # Create labels with team color indicator
                x_labels = []
                y_labels = []
                for i, (pid, team) in enumerate(zip(player_ids, player_teams)):
                    team_marker = "🔴" if team == 0 else "🔵"
                    x_labels.append(f"{team_marker} {pid}")
                    y_labels.append(f"{team_marker} {pid}")

                ax.set_xticklabels(x_labels, rotation=90)
                ax.set_yticklabels(y_labels)

                # Add grid lines to separate cells
                ax.set_xticks(np.arange(-0.5, n, 1), minor=True)
                ax.set_yticks(np.arange(-0.5, n, 1), minor=True)
                ax.grid(which="minor", color="w", linestyle="-", linewidth=2)

                # Add team separation lines
                team_changes = [
                    i for i in range(1, n) if player_teams[i] != player_teams[i - 1]
                ]
                for pos in team_changes:
                    ax.axhline(y=pos - 0.5, color="white", linewidth=4)
                    ax.axvline(x=pos - 0.5, color="white", linewidth=4)

                # Add value annotations in cells
                for i in range(n):
                    for j in range(n):
                        if matrix[i, j] > 0:
                            text_color = (
                                "white"
                                if matrix[i, j] > np.max(matrix) / 2
                                else "black"
                            )
                            ax.text(
                                j,
                                i,
                                f"{int(matrix[i, j])}",
                                ha="center",
                                va="center",
                                color=text_color,
                            )

                ax.set_title("Player Interaction Matrix")
                matrix_fig.tight_layout()
                matrix_canvas.draw()

            except Exception as e:
                messagebox.showerror(
                    "Error", f"Error updating matrix visualization: {str(e)}"
                )

        # Function to update team interactions visualization
        def update_team_interactions():
            try:
                # Clear figure
                team_fig.clear()

                # Create two subplots
                ax1 = team_fig.add_subplot(121)
                ax2 = team_fig.add_subplot(122)

                # Plot team interaction counts
                if team_interactions:
                    # Extract data
                    labels = ["Team 1 Internal", "Team 2 Internal", "Cross-Team"]
                    values = [
                        team_interactions.get((0, 0), 0),
                        team_interactions.get((1, 1), 0),
                        team_interactions.get((0, 1), 0),
                    ]
                    colors = ["red", "blue", "purple"]

                    # Pie chart
                    ax1.pie(
                        values,
                        labels=labels,
                        colors=colors,
                        autopct="%1.1f%%",
                        startangle=90,
                    )
                    ax1.axis("equal")
                    ax1.set_title("Interaction Distribution")

                    # Bar chart
                    ax2.bar(labels, values, color=colors)
                    ax2.set_title("Interaction Counts")
                    ax2.set_ylabel("Number of Interactions")
                    ax2.tick_params(axis="x", rotation=45)

                    # Add count labels on bars
                    for i, v in enumerate(values):
                        ax2.text(i, v + 0.1, str(v), ha="center", va="bottom")
                else:
                    ax1.text(
                        0.5,
                        0.5,
                        "No team interaction data available",
                        horizontalalignment="center",
                        verticalalignment="center",
                        transform=ax1.transAxes,
                    )
                    ax2.text(
                        0.5,
                        0.5,
                        "No team interaction data available",
                        horizontalalignment="center",
                        verticalalignment="center",
                        transform=ax2.transAxes,
                    )

                team_fig.tight_layout()
                team_canvas.draw()

            except Exception as e:
                messagebox.showerror(
                    "Error", f"Error updating team visualization: {str(e)}"
                )

        # Function to update stats visualization
        def update_stats():
            try:
                # Clear previous content
                for widget in stats_frame.winfo_children():
                    widget.destroy()

                # Create stats content
                header_frame = ttk.Frame(stats_frame)
                header_frame.pack(fill=tk.X, pady=(0, 10))

                ttk.Label(
                    header_frame,
                    text="Interaction Statistics",
                    font=("Arial", 16, "bold"),
                ).pack()

                # Create scrollable frame
                canvas = tk.Canvas(stats_frame)
                scrollbar = ttk.Scrollbar(
                    stats_frame, orient="vertical", command=canvas.yview
                )
                scrollable_frame = ttk.Frame(canvas)

                scrollable_frame.bind(
                    "<Configure>",
                    lambda e: canvas.configure(scrollregion=canvas.bbox("all")),
                )

                canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
                canvas.configure(yscrollcommand=scrollbar.set)

                canvas.pack(side="left", fill="both", expand=True)
                scrollbar.pack(side="right", fill="y")

                # Overall stats
                stats_box = ttk.LabelFrame(scrollable_frame, text="Overall Statistics")
                stats_box.pack(fill=tk.X, pady=10, padx=5)

                total_interactions = sum(player_interactions.values())
                unique_pairs = len(player_interactions)

                if team_interactions:
                    team1_internal = team_interactions.get((0, 0), 0)
                    team2_internal = team_interactions.get((1, 1), 0)
                    cross_team = team_interactions.get((0, 1), 0)

                    team1_pct = (
                        (team1_internal / total_interactions * 100)
                        if total_interactions > 0
                        else 0
                    )
                    team2_pct = (
                        (team2_internal / total_interactions * 100)
                        if total_interactions > 0
                        else 0
                    )
                    cross_pct = (
                        (cross_team / total_interactions * 100)
                        if total_interactions > 0
                        else 0
                    )
                else:
                    team1_internal = team2_internal = cross_team = 0
                    team1_pct = team2_pct = cross_pct = 0

                # Create grid for stats
                stats_grid = ttk.Frame(stats_box)
                stats_grid.pack(fill=tk.X, padx=10, pady=10)

                row = 0
                ttk.Label(stats_grid, text="Total Interactions:").grid(
                    row=row, column=0, sticky=tk.W, pady=2
                )
                ttk.Label(stats_grid, text=str(total_interactions)).grid(
                    row=row, column=1, sticky=tk.W, pady=2
                )

                row += 1
                ttk.Label(stats_grid, text="Unique Player Pairs:").grid(
                    row=row, column=0, sticky=tk.W, pady=2
                )
                ttk.Label(stats_grid, text=str(unique_pairs)).grid(
                    row=row, column=1, sticky=tk.W, pady=2
                )

                row += 1
                ttk.Label(stats_grid, text="Team 1 Internal:").grid(
                    row=row, column=0, sticky=tk.W, pady=2
                )
                ttk.Label(stats_grid, text=f"{team1_internal} ({team1_pct:.1f}%)").grid(
                    row=row, column=1, sticky=tk.W, pady=2
                )

                row += 1
                ttk.Label(stats_grid, text="Team 2 Internal:").grid(
                    row=row, column=0, sticky=tk.W, pady=2
                )
                ttk.Label(stats_grid, text=f"{team2_internal} ({team2_pct:.1f}%)").grid(
                    row=row, column=1, sticky=tk.W, pady=2
                )

                row += 1
                ttk.Label(stats_grid, text="Cross-Team:").grid(
                    row=row, column=0, sticky=tk.W, pady=2
                )
                ttk.Label(stats_grid, text=f"{cross_team} ({cross_pct:.1f}%)").grid(
                    row=row, column=1, sticky=tk.W, pady=2
                )

                # Player interaction stats
                player_stats_box = ttk.LabelFrame(
                    scrollable_frame, text="Player Interaction Statistics"
                )
                player_stats_box.pack(fill=tk.X, pady=10, padx=5)

                # Calculate interaction counts per player
                player_interaction_counts = defaultdict(int)
                for (p1, p2), count in player_interactions.items():
                    player_interaction_counts[p1] += count
                    player_interaction_counts[p2] += count

                # Sort players by interaction count
                sorted_players = sorted(
                    player_interaction_counts.items(), key=lambda x: x[1], reverse=True
                )

                # Create player stats grid
                player_grid = ttk.Frame(player_stats_box)
                player_grid.pack(fill=tk.X, padx=10, pady=10)

                # Header row
                columns = [
                    "Rank",
                    "Player",
                    "Team",
                    "Total Interactions",
                    "% of Interactions",
                ]
                for col, text in enumerate(columns):
                    ttk.Label(player_grid, text=text, font=("Arial", 10, "bold")).grid(
                        row=0, column=col, sticky=tk.W, padx=5, pady=5
                    )

                # Add player rows
                for rank, (player_id, count) in enumerate(sorted_players[:20], 1):
                    team = self.tracker.player_teams.get(player_id, -1)
                    team_str = f"Team {team+1}" if team != -1 else "Unknown"

                    pct = (
                        (count / (total_interactions * 2) * 100)
                        if total_interactions > 0
                        else 0
                    )

                    row = rank
                    col = 0

                    ttk.Label(player_grid, text=str(rank)).grid(
                        row=row, column=col, sticky=tk.W, padx=5, pady=2
                    )
                    col += 1

                    player_label = ttk.Label(player_grid, text=f"Player {player_id}")
                    if team == 0:
                        player_label.configure(foreground="red")
                    elif team == 1:
                        player_label.configure(foreground="blue")
                    player_label.grid(row=row, column=col, sticky=tk.W, padx=5, pady=2)
                    col += 1

                    ttk.Label(player_grid, text=team_str).grid(
                        row=row, column=col, sticky=tk.W, padx=5, pady=2
                    )
                    col += 1

                    ttk.Label(player_grid, text=str(count)).grid(
                        row=row, column=col, sticky=tk.W, padx=5, pady=2
                    )
                    col += 1

                    ttk.Label(player_grid, text=f"{pct:.1f}%").grid(
                        row=row, column=col, sticky=tk.W, padx=5, pady=2
                    )

                # Pair interaction stats
                pair_stats_box = ttk.LabelFrame(
                    scrollable_frame, text="Top Player Pairs"
                )
                pair_stats_box.pack(fill=tk.X, pady=10, padx=5)

                # Sort pairs by interaction count
                sorted_pairs = sorted(
                    player_interactions.items(), key=lambda x: x[1], reverse=True
                )

                # Create pair stats grid
                pair_grid = ttk.Frame(pair_stats_box)
                pair_grid.pack(fill=tk.X, padx=10, pady=10)

                # Header row
                columns = [
                    "Rank",
                    "Player Pair",
                    "Interaction Type",
                    "Interaction Count",
                    "% of Total",
                ]
                for col, text in enumerate(columns):
                    ttk.Label(pair_grid, text=text, font=("Arial", 10, "bold")).grid(
                        row=0, column=col, sticky=tk.W, padx=5, pady=5
                    )

                # Add pair rows
                for rank, ((p1, p2), count) in enumerate(sorted_pairs[:15], 1):
                    team1 = self.tracker.player_teams.get(p1, -1)
                    team2 = self.tracker.player_teams.get(p2, -1)

                    if team1 == team2:
                        if team1 == 0:
                            interaction_type = "Team 1 Internal"
                            color = "red"
                        elif team1 == 1:
                            interaction_type = "Team 2 Internal"
                            color = "blue"
                        else:
                            interaction_type = "Unknown"
                            color = "black"
                    else:
                        interaction_type = "Cross-Team"
                        color = "purple"

                    pct = (
                        (count / total_interactions * 100)
                        if total_interactions > 0
                        else 0
                    )

                    row = rank
                    col = 0

                    ttk.Label(pair_grid, text=str(rank)).grid(
                        row=row, column=col, sticky=tk.W, padx=5, pady=2
                    )
                    col += 1

                    pair_label = ttk.Label(pair_grid, text=f"Player {p1} — Player {p2}")
                    pair_label.configure(foreground=color)
                    pair_label.grid(row=row, column=col, sticky=tk.W, padx=5, pady=2)
                    col += 1

                    ttk.Label(pair_grid, text=interaction_type).grid(
                        row=row, column=col, sticky=tk.W, padx=5, pady=2
                    )
                    col += 1

                    ttk.Label(pair_grid, text=str(count)).grid(
                        row=row, column=col, sticky=tk.W, padx=5, pady=2
                    )
                    col += 1

                    ttk.Label(pair_grid, text=f"{pct:.1f}%").grid(
                        row=row, column=col, sticky=tk.W, padx=5, pady=2
                    )

            except Exception as e:
                messagebox.showerror("Error", f"Error updating stats: {str(e)}")

        # Initial updates
        update_network()

    def show_advanced_settings(self):
        """Show advanced application settings"""
        # Create dialog
        dialog = tk.Toplevel(self.root)
        dialog.title("Advanced Settings")
        dialog.geometry("600x700")
        dialog.transient(self.root)

        # Create main frame with notebook
        notebook = ttk.Notebook(dialog)
        notebook.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        # General settings tab
        general_tab = ttk.Frame(notebook)
        notebook.add(general_tab, text="General")

        # Detection settings tab
        detection_tab = ttk.Frame(notebook)
        notebook.add(detection_tab, text="Detection")

        # Analysis settings tab
        analysis_tab = ttk.Frame(notebook)
        notebook.add(analysis_tab, text="Analysis")

        # Visualization settings tab
        vis_tab = ttk.Frame(notebook)
        notebook.add(vis_tab, text="Visualization")

        # Camera settings tab
        camera_tab = ttk.Frame(notebook)
        notebook.add(camera_tab, text="Camera")

        # General settings content
        general_frame = ttk.Frame(general_tab, padding=10)
        general_frame.pack(fill=tk.BOTH, expand=True)

        # Output directory
        dir_frame = ttk.Frame(general_frame)
        dir_frame.pack(fill=tk.X, pady=10)

        ttk.Label(dir_frame, text="Output Directory:").grid(
            row=0, column=0, sticky=tk.W, padx=5
        )

        output_dir_var = tk.StringVar(value=CONFIG["output_dir"])
        output_entry = ttk.Entry(dir_frame, textvariable=output_dir_var, width=40)
        output_entry.grid(row=0, column=1, sticky=tk.W + tk.E, padx=5)

        ttk.Button(
            dir_frame,
            text="Browse...",
            command=lambda: output_dir_var.set(
                filedialog.askdirectory(initialdir=output_dir_var.get())
            ),
        ).grid(row=0, column=2, padx=5)

        # Max processing FPS
        fps_frame = ttk.Frame(general_frame)
        fps_frame.pack(fill=tk.X, pady=10)

        ttk.Label(fps_frame, text="Max Processing FPS:").grid(
            row=0, column=0, sticky=tk.W, padx=5
        )

        fps_var = tk.IntVar(value=CONFIG["max_fps_processing"])
        fps_spinbox = ttk.Spinbox(
            fps_frame, from_=1, to=60, textvariable=fps_var, width=5
        )
        fps_spinbox.grid(row=0, column=1, sticky=tk.W, padx=5)

        # UI theme
        theme_frame = ttk.Frame(general_frame)
        theme_frame.pack(fill=tk.X, pady=10)

        ttk.Label(theme_frame, text="UI Theme:").grid(
            row=0, column=0, sticky=tk.W, padx=5
        )

        theme_var = tk.StringVar(value=CONFIG["default_ui_theme"])
        theme_combo = ttk.Combobox(
            theme_frame,
            textvariable=theme_var,
            values=["light", "dark"],
            state="readonly",
            width=10,
        )
        theme_combo.grid(row=0, column=1, sticky=tk.W, padx=5)

        # Detection settings content
        detection_frame = ttk.Frame(detection_tab, padding=10)
        detection_frame.pack(fill=tk.BOTH, expand=True)

        # Confidence threshold
        conf_frame = ttk.Frame(detection_frame)
        conf_frame.pack(fill=tk.X, pady=10)

        ttk.Label(conf_frame, text="Confidence Threshold:").grid(
            row=0, column=0, sticky=tk.W, padx=5
        )

        conf_var = tk.DoubleVar(value=CONFIG["confidence_threshold"])
        conf_scale = ttk.Scale(
            conf_frame, from_=0.1, to=0.9, variable=conf_var, orient=tk.HORIZONTAL
        )
        conf_scale.grid(row=0, column=1, sticky=tk.W + tk.E, padx=5)

        conf_label = ttk.Label(conf_frame, text=f"{conf_var.get():.1f}")
        conf_label.grid(row=0, column=2, padx=5)

        # Update conf label when slider changes
        def update_conf_label(*args):
            conf_label.configure(text=f"{conf_var.get():.1f}")

        conf_var.trace("w", update_conf_label)

        # Model path
        model_frame = ttk.Frame(detection_frame)
        model_frame.pack(fill=tk.X, pady=10)

        ttk.Label(model_frame, text="YOLO Model Path:").grid(
            row=0, column=0, sticky=tk.W, padx=5
        )

        model_var = tk.StringVar(value=CONFIG["model_path"])
        model_entry = ttk.Entry(model_frame, textvariable=model_var, width=40)
        model_entry.grid(row=0, column=1, sticky=tk.W + tk.E, padx=5)

        ttk.Button(
            model_frame,
            text="Browse...",
            command=lambda: model_var.set(
                filedialog.askopenfilename(
                    initialdir=os.path.dirname(model_var.get()),
                    filetypes=[("PT files", "*.pt"), ("All files", "*.*")],
                )
            ),
        ).grid(row=0, column=2, padx=5)

        # Advanced settings
        adv_frame = ttk.LabelFrame(detection_frame, text="Advanced Detection Settings")
        adv_frame.pack(fill=tk.X, pady=10)

        # Team detection
        team_var = tk.BooleanVar(
            value=CONFIG["advanced_analysis"]["formation_detection"]
        )
        ttk.Checkbutton(
            adv_frame, text="Enable Team Detection", variable=team_var
        ).pack(anchor=tk.W, padx=10, pady=5)

        # Action detection
        action_var = tk.BooleanVar(value=CONFIG["advanced_analysis"]["event_detection"])
        ttk.Checkbutton(
            adv_frame, text="Enable Action Detection", variable=action_var
        ).pack(anchor=tk.W, padx=10, pady=5)

        # Analysis settings content
        analysis_frame = ttk.Frame(analysis_tab, padding=10)
        analysis_frame.pack(fill=tk.BOTH, expand=True)

        # Advanced analysis options
        analysis_settings = ttk.LabelFrame(analysis_frame, text="Analysis Features")
        analysis_settings.pack(fill=tk.X, pady=10)

        # Formation detection
        formation_var = tk.BooleanVar(
            value=CONFIG["advanced_analysis"]["formation_detection"]
        )
        ttk.Checkbutton(
            analysis_settings, text="Formation Detection", variable=formation_var
        ).pack(anchor=tk.W, padx=10, pady=5)

        # Possession analysis
        possession_var = tk.BooleanVar(
            value=CONFIG["advanced_analysis"]["possession_analysis"]
        )
        ttk.Checkbutton(
            analysis_settings, text="Possession Analysis", variable=possession_var
        ).pack(anchor=tk.W, padx=10, pady=5)

        # Team shape analysis
        shape_var = tk.BooleanVar(
            value=CONFIG["advanced_analysis"]["team_shape_analysis"]
        )
        ttk.Checkbutton(
            analysis_settings, text="Team Shape Analysis", variable=shape_var
        ).pack(anchor=tk.W, padx=10, pady=5)

        # Player role detection
        role_var = tk.BooleanVar(
            value=CONFIG["advanced_analysis"]["player_role_detection"]
        )
        ttk.Checkbutton(
            analysis_settings, text="Player Role Detection", variable=role_var
        ).pack(anchor=tk.W, padx=10, pady=5)

        # Visualization settings content
        vis_frame = ttk.Frame(vis_tab, padding=10)
        vis_frame.pack(fill=tk.BOTH, expand=True)

        # Colormap
        colormap_frame = ttk.Frame(vis_frame)
        colormap_frame.pack(fill=tk.X, pady=10)

        ttk.Label(colormap_frame, text="Default Colormap:").grid(
            row=0, column=0, sticky=tk.W, padx=5
        )

        colormap_var = tk.StringVar(
            value=CONFIG["visualization_settings"]["default_colormap"]
        )
        colormap_combo = ttk.Combobox(
            colormap_frame,
            textvariable=colormap_var,
            values=[
                "viridis",
                "plasma",
                "magma",
                "inferno",
                "cividis",
                "Reds",
                "Blues",
                "Greens",
                "YlOrRd",
                "coolwarm",
            ],
            state="readonly",
            width=15,
        )
        colormap_combo.grid(row=0, column=1, sticky=tk.W, padx=5)

        # Field overlay opacity
        opacity_frame = ttk.Frame(vis_frame)
        opacity_frame.pack(fill=tk.X, pady=10)

        ttk.Label(opacity_frame, text="Field Overlay Opacity:").grid(
            row=0, column=0, sticky=tk.W, padx=5
        )

        opacity_var = tk.DoubleVar(
            value=CONFIG["visualization_settings"]["field_overlay_opacity"]
        )
        opacity_scale = ttk.Scale(
            opacity_frame, from_=0.1, to=1.0, variable=opacity_var, orient=tk.HORIZONTAL
        )
        opacity_scale.grid(row=0, column=1, sticky=tk.W + tk.E, padx=5)

        opacity_label = ttk.Label(opacity_frame, text=f"{opacity_var.get():.1f}")
        opacity_label.grid(row=0, column=2, padx=5)

        # Update opacity label when slider changes
        def update_opacity_label(*args):
            opacity_label.configure(text=f"{opacity_var.get():.1f}")

        opacity_var.trace("w", update_opacity_label)

        # Marker size
        marker_frame = ttk.Frame(vis_frame)
        marker_frame.pack(fill=tk.X, pady=10)

        ttk.Label(marker_frame, text="Marker Size:").grid(
            row=0, column=0, sticky=tk.W, padx=5
        )

        marker_var = tk.IntVar(value=CONFIG["visualization_settings"]["marker_size"])
        marker_spinbox = ttk.Spinbox(
            marker_frame, from_=4, to=20, textvariable=marker_var, width=5
        )
        marker_spinbox.grid(row=0, column=1, sticky=tk.W, padx=5)

        # Line width
        line_frame = ttk.Frame(vis_frame)
        line_frame.pack(fill=tk.X, pady=10)

        ttk.Label(line_frame, text="Trajectory Line Width:").grid(
            row=0, column=0, sticky=tk.W, padx=5
        )

        line_var = tk.IntVar(
            value=CONFIG["visualization_settings"]["trajectory_line_width"]
        )
        line_spinbox = ttk.Spinbox(
            line_frame, from_=1, to=10, textvariable=line_var, width=5
        )
        line_spinbox.grid(row=0, column=1, sticky=tk.W, padx=5)

        # Camera settings content
        camera_frame = ttk.Frame(camera_tab, padding=10)
        camera_frame.pack(fill=tk.BOTH, expand=True)

        # Webcam settings
        webcam_settings = ttk.LabelFrame(camera_frame, text="Webcam Settings")
        webcam_settings.pack(fill=tk.X, pady=10)

        # Default device
        device_frame = ttk.Frame(webcam_settings)
        device_frame.pack(fill=tk.X, padx=10, pady=5)

        ttk.Label(device_frame, text="Default Device:").grid(
            row=0, column=0, sticky=tk.W, padx=5
        )

        device_var = tk.IntVar(
            value=CONFIG["camera_settings"]["webcam"]["default_device"]
        )
        device_spinbox = ttk.Spinbox(
            device_frame, from_=0, to=10, textvariable=device_var, width=5
        )
        device_spinbox.grid(row=0, column=1, sticky=tk.W, padx=5)

        # Resolution
        res_frame = ttk.Frame(webcam_settings)
        res_frame.pack(fill=tk.X, padx=10, pady=5)

        ttk.Label(res_frame, text="Resolution:").grid(
            row=0, column=0, sticky=tk.W, padx=5
        )

        res_w, res_h = CONFIG["camera_settings"]["webcam"]["resolution"]
        res_w_var = tk.IntVar(value=res_w)
        res_h_var = tk.IntVar(value=res_h)

        res_w_spinbox = ttk.Spinbox(
            res_frame, from_=320, to=3840, textvariable=res_w_var, width=5
        )
        res_w_spinbox.grid(row=0, column=1, sticky=tk.W, padx=5)

        ttk.Label(res_frame, text="×").grid(row=0, column=2, padx=5)

        res_h_spinbox = ttk.Spinbox(
            res_frame, from_=240, to=2160, textvariable=res_h_var, width=5
        )
        res_h_spinbox.grid(row=0, column=3, sticky=tk.W, padx=5)

        # IP camera settings
        ip_settings = ttk.LabelFrame(camera_frame, text="IP Camera Settings")
        ip_settings.pack(fill=tk.X, pady=10)

        # Default URL
        url_frame = ttk.Frame(ip_settings)
        url_frame.pack(fill=tk.X, padx=10, pady=5)

        ttk.Label(url_frame, text="Default URL:").grid(
            row=0, column=0, sticky=tk.W, padx=5
        )

        url_var = tk.StringVar(
            value=CONFIG["camera_settings"]["ip_camera"]["default_url"]
        )
        url_entry = ttk.Entry(url_frame, textvariable=url_var, width=40)
        url_entry.grid(row=0, column=1, sticky=tk.W + tk.E, padx=5, columnspan=3)

        # Default credentials
        cred_frame = ttk.Frame(ip_settings)
        cred_frame.pack(fill=tk.X, padx=10, pady=5)

        ttk.Label(cred_frame, text="Default Username:").grid(
            row=0, column=0, sticky=tk.W, padx=5
        )

        username_var = tk.StringVar(
            value=CONFIG["camera_settings"]["ip_camera"]["username"]
        )
        username_entry = ttk.Entry(cred_frame, textvariable=username_var, width=15)
        username_entry.grid(row=0, column=1, sticky=tk.W, padx=5)

        ttk.Label(cred_frame, text="Password:").grid(
            row=0, column=2, sticky=tk.W, padx=5
        )

        password_var = tk.StringVar(
            value=CONFIG["camera_settings"]["ip_camera"]["password"]
        )
        password_entry = ttk.Entry(
            cred_frame, textvariable=password_var, width=15, show="*"
        )
        password_entry.grid(row=0, column=3, sticky=tk.W, padx=5)

        # Buttons
        button_frame = ttk.Frame(dialog)
        button_frame.pack(fill=tk.X, pady=10, padx=10)

        def save_settings():
            try:
                # Update CONFIG dictionary with new values
                # General settings
                CONFIG["output_dir"] = output_dir_var.get()
                CONFIG["max_fps_processing"] = fps_var.get()
                CONFIG["default_ui_theme"] = theme_var.get()

                # Detection settings
                CONFIG["confidence_threshold"] = conf_var.get()
                CONFIG["model_path"] = model_var.get()

                # Advanced analysis settings
                CONFIG["advanced_analysis"]["formation_detection"] = formation_var.get()
                CONFIG["advanced_analysis"][
                    "possession_analysis"
                ] = possession_var.get()
                CONFIG["advanced_analysis"]["team_shape_analysis"] = shape_var.get()
                CONFIG["advanced_analysis"]["player_role_detection"] = role_var.get()
                CONFIG["advanced_analysis"]["event_detection"] = action_var.get()

                # Visualization settings
                CONFIG["visualization_settings"][
                    "default_colormap"
                ] = colormap_var.get()
                CONFIG["visualization_settings"][
                    "field_overlay_opacity"
                ] = opacity_var.get()
                CONFIG["visualization_settings"]["marker_size"] = marker_var.get()
                CONFIG["visualization_settings"][
                    "trajectory_line_width"
                ] = line_var.get()

                # Camera settings
                CONFIG["camera_settings"]["webcam"]["default_device"] = device_var.get()
                CONFIG["camera_settings"]["webcam"]["resolution"] = (
                    res_w_var.get(),
                    res_h_var.get(),
                )
                CONFIG["camera_settings"]["ip_camera"]["default_url"] = url_var.get()
                CONFIG["camera_settings"]["ip_camera"]["username"] = username_var.get()
                CONFIG["camera_settings"]["ip_camera"]["password"] = password_var.get()

                # Create output directories
                os.makedirs(CONFIG["output_dir"], exist_ok=True)
                os.makedirs(os.path.join(CONFIG["output_dir"], "videos"), exist_ok=True)
                os.makedirs(os.path.join(CONFIG["output_dir"], "data"), exist_ok=True)
                os.makedirs(
                    os.path.join(CONFIG["output_dir"], "visualizations"), exist_ok=True
                )
                os.makedirs(
                    os.path.join(CONFIG["output_dir"], "reports"), exist_ok=True
                )

                # Apply theme if changed
                if theme_var.get() != self.theme:
                    self.theme = theme_var.get()
                    self.set_theme(self.theme)

                messagebox.showinfo(
                    "Settings Saved", "Settings have been updated successfully."
                )
                dialog.destroy()

            except Exception as e:
                messagebox.showerror("Error", f"Error saving settings: {str(e)}")

        ttk.Button(button_frame, text="Save", command=save_settings).pack(
            side=tk.RIGHT, padx=5
        )
        ttk.Button(button_frame, text="Cancel", command=dialog.destroy).pack(
            side=tk.RIGHT, padx=5
        )

    def show_documentation(self):
        """Show application documentation"""
        # Create dialog
        dialog = tk.Toplevel(self.root)
        dialog.title("Documentation")
        dialog.geometry("800x600")
        dialog.transient(self.root)

        # Create main frame with notebook
        notebook = ttk.Notebook(dialog)
        notebook.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        # Overview tab
        overview_tab = ttk.Frame(notebook)
        notebook.add(overview_tab, text="Overview")

        # Getting Started tab
        started_tab = ttk.Frame(notebook)
        notebook.add(started_tab, text="Getting Started")

        # Features tab
        features_tab = ttk.Frame(notebook)
        notebook.add(features_tab, text="Features")

        # Visualization tab
        vis_tab = ttk.Frame(notebook)
        notebook.add(vis_tab, text="Visualization")

        # Troubleshooting tab
        trouble_tab = ttk.Frame(notebook)
        notebook.add(trouble_tab, text="Troubleshooting")

        # Overview content
        overview_frame = ttk.Frame(overview_tab, padding=20)
        overview_frame.pack(fill=tk.BOTH, expand=True)

        ttk.Label(
            overview_frame,
            text="AI-Powered Sports Analytics & Player Tracking",
            font=("Arial", 16, "bold"),
        ).pack(anchor=tk.W, pady=(0, 20))

        overview_text = """
This application provides advanced sports analytics and player tracking capabilities using 
computer vision and machine learning. Built with YOLOv8 and OpenCV, it enables:

- Automated player tracking with speed, distance, and movement analysis
- Team and player identification
- Action and event detection
- Formation and tactical analysis
- Comprehensive visualization tools
- Real-time analysis from various video sources

The application supports multiple sports including football (soccer), basketball, 
cricket, volleyball, and hockey.
        """

        ttk.Label(
            overview_frame, text=overview_text, wraplength=700, justify=tk.LEFT
        ).pack(fill=tk.X)

        # System Requirements
        ttk.Label(
            overview_frame, text="System Requirements", font=("Arial", 12, "bold")
        ).pack(anchor=tk.W, pady=(20, 10))

        requirements_text = """
- Python 3.8 or higher
- GPU recommended for real-time analysis (with CUDA support)
- Minimum 8GB RAM (16GB recommended)
- At least 2GB of free disk space
- Dependencies: OpenCV, PyTorch, Ultralytics, Matplotlib, Plotly, NetworkX, SciPy, Pandas
        """

        ttk.Label(
            overview_frame, text=requirements_text, wraplength=700, justify=tk.LEFT
        ).pack(fill=tk.X)

        # Getting Started content
        started_frame = ttk.Frame(started_tab, padding=20)
        started_frame.pack(fill=tk.BOTH, expand=True)

        ttk.Label(
            started_frame, text="Getting Started", font=("Arial", 16, "bold")
        ).pack(anchor=tk.W, pady=(0, 20))

        # Create canvas with scrollbar
        canvas = tk.Canvas(started_frame)
        scrollbar = ttk.Scrollbar(
            started_frame, orient="vertical", command=canvas.yview
        )
        scrollable_frame = ttk.Frame(canvas)

        scrollable_frame.bind(
            "<Configure>", lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
        )

        canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)

        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")

        # Add content to scrollable frame
        ttk.Label(
            scrollable_frame, text="1. Loading a Video", font=("Arial", 12, "bold")
        ).pack(anchor=tk.W, pady=(0, 10))

        video_text = """
To analyze a sports video:
1. Click on "File > Open Video" or use the "Video File" button in the main interface
2. Select the video file from your computer
3. Choose the appropriate sport from the dropdown menu
4. Adjust detection settings if needed
5. Click "Start Processing" to begin analysis
        """

        ttk.Label(
            scrollable_frame, text=video_text, wraplength=700, justify=tk.LEFT
        ).pack(fill=tk.X)

        ttk.Label(
            scrollable_frame, text="2. Using Live Camera", font=("Arial", 12, "bold")
        ).pack(anchor=tk.W, pady=(20, 10))

        camera_text = """
For real-time analysis:
1. Select "Video Source > Webcam" or "Video Source > IP Camera"
2. Configure the camera settings
3. Select the sport and adjust detection settings
4. Click "Start Processing" to begin real-time analysis
        """

        ttk.Label(
            scrollable_frame, text=camera_text, wraplength=700, justify=tk.LEFT
        ).pack(fill=tk.X)

        ttk.Label(
            scrollable_frame, text="3. Viewing Results", font=("Arial", 12, "bold")
        ).pack(anchor=tk.W, pady=(20, 10))

        results_text = """
After processing:
1. The "Results" panel shows detected players and their statistics
2. Use the visualization tools in the right panel to explore the data:
   • Player Stats: View detailed statistics for individual players
   • Team Analysis: Analyze team performance and patterns
   • Heatmaps: See position density and movement patterns
   • Trajectories: Visualize player movement paths
   • Speed Profiles: Analyze speed patterns and performance
   • Interaction Network: See how players interact on the field
3. Generate a comprehensive report using "Generate Report" button
        """

        ttk.Label(
            scrollable_frame, text=results_text, wraplength=700, justify=tk.LEFT
        ).pack(fill=tk.X)

        ttk.Label(
            scrollable_frame, text="4. Saving and Exporting", font=("Arial", 12, "bold")
        ).pack(anchor=tk.W, pady=(20, 10))

        export_text = """
To save your analysis:
1. Click "Save Results" to save the tracking data for later use
2. Use "Export Data" to export in various formats (CSV, JSON, Excel)
3. Generate a report with "Generate Report" to create an HTML report with visualizations
4. The processed video with annotations can be saved during initial processing
        """

        ttk.Label(
            scrollable_frame, text=export_text, wraplength=700, justify=tk.LEFT
        ).pack(fill=tk.X)

        # Features content
        features_frame = ttk.Frame(features_tab, padding=20)
        features_frame.pack(fill=tk.BOTH, expand=True)

        ttk.Label(features_frame, text="Key Features", font=("Arial", 16, "bold")).pack(
            anchor=tk.W, pady=(0, 20)
        )

        # Create scrollable content
        features_canvas = tk.Canvas(features_frame)
        features_scrollbar = ttk.Scrollbar(
            features_frame, orient="vertical", command=features_canvas.yview
        )
        features_scrollable = ttk.Frame(features_canvas)

        features_scrollable.bind(
            "<Configure>",
            lambda e: features_canvas.configure(
                scrollregion=features_canvas.bbox("all")
            ),
        )

        features_canvas.create_window((0, 0), window=features_scrollable, anchor="nw")
        features_canvas.configure(yscrollcommand=features_scrollbar.set)

        features_canvas.pack(side="left", fill="both", expand=True)
        features_scrollbar.pack(side="right", fill="y")

        # Feature list
        features = [
            {
                "title": "Player Tracking",
                "desc": "Track player positions throughout the game with high accuracy. The system automatically assigns IDs to players and maintains their identity across frames.",
                "sub_features": [
                    "Accurate bounding box detection using YOLOv8",
                    "Persistent tracking across frames with occlusion handling",
                    "Team classification for team sports",
                    "Position mapping to standardized field coordinates",
                ],
            },
            {
                "title": "Movement Analytics",
                "desc": "Analyze player movement patterns, speeds, and distances to gain insights into physical performance and tactical positioning.",
                "sub_features": [
                    "Distance covered measurement",
                    "Speed profile analysis (avg, max, distribution)",
                    "Acceleration and deceleration detection",
                    "Zone-based movement analysis",
                ],
            },
            {
                "title": "Action Detection",
                "desc": "Automatically detect key actions and events during gameplay to create a timeline of significant moments.",
                "sub_features": [
                    f"Sport-specific action detection ({', '.join(CONFIG['sports']['football']['actions'][:3])} etc. for football)",
                    "Event timeline generation",
                    "Action frequency analysis",
                    "Player-specific action detection",
                ],
            },
            {
                "title": "Team Analysis",
                "desc": "Analyze team patterns, formations, and tactical approaches through advanced spatial analysis.",
                "sub_features": [
                    "Team formation detection and tracking",
                    "Possession analysis by time and zone",
                    "Team shape and structure analysis",
                    "Inter-team and intra-team interaction analysis",
                ],
            },
            {
                "title": "Visualization Tools",
                "desc": "Powerful visualization tools to present analytics in an intuitive and informative way.",
                "sub_features": [
                    "Heatmaps for position density",
                    "Trajectory visualizations with time progression",
                    "Speed and performance charts",
                    "Network graphs for player interactions",
                ],
            },
        ]

        # Add features to scrollable frame
        for i, feature in enumerate(features):
            ttk.Label(
                features_scrollable, text=feature["title"], font=("Arial", 12, "bold")
            ).pack(anchor=tk.W, pady=(20 if i > 0 else 0, 10))

            ttk.Label(
                features_scrollable,
                text=feature["desc"],
                wraplength=700,
                justify=tk.LEFT,
            ).pack(fill=tk.X)

            # Add sub-features
            sub_frame = ttk.Frame(features_scrollable)
            sub_frame.pack(fill=tk.X, padx=20, pady=5)

            for j, sub in enumerate(feature["sub_features"]):
                ttk.Label(
                    sub_frame, text=f"• {sub}", wraplength=650, justify=tk.LEFT
                ).pack(anchor=tk.W, pady=2)

        # Button frame with close button
        button_frame = ttk.Frame(dialog)
        button_frame.pack(fill=tk.X, pady=10, padx=10)

        ttk.Button(button_frame, text="Close", command=dialog.destroy).pack(
            side=tk.RIGHT
        )

    def show_about(self):
        """Show about dialog"""
        dialog = tk.Toplevel(self.root)
        dialog.title("About")
        dialog.geometry("400x300")
        dialog.transient(self.root)

        # Create main frame
        main_frame = ttk.Frame(dialog, padding=20)
        main_frame.pack(fill=tk.BOTH, expand=True)

        ttk.Label(
            main_frame, text="AI-Powered Sports Analytics", font=("Arial", 16, "bold")
        ).pack(pady=(0, 5))

        ttk.Label(main_frame, text="Version 1.0.0").pack(pady=(0, 20))

        about_text = """
A comprehensive sports analytics and player tracking application 
powered by computer vision and machine learning.

Built with YOLOv8, OpenCV, and Python.

© 2025 Sports Analytics Team
        """

        ttk.Label(main_frame, text=about_text, justify=tk.CENTER).pack(pady=10)

        ttk.Button(main_frame, text="Close", command=dialog.destroy).pack(pady=20)


def main():
    """Main function to run the application"""
    root = tk.Tk()
    root.title("AI-Powered Sports Analytics & Player Tracking")

    # Set icon if available
    try:
        icon_path = os.path.join(os.path.dirname(__file__), "icon.ico")
        if os.path.exists(icon_path):
            root.iconbitmap(icon_path)
    except:
        pass

    # Create and run application
    app = SportsAnalyticsGUI(root)
    root.mainloop()


if __name__ == "__main__":
    main()
