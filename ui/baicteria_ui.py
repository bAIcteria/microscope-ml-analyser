import sys
import os
from PyQt5.QtWidgets import (
    QApplication, QWidget, QVBoxLayout, QPushButton, QLabel, QHBoxLayout,
    QFileDialog, QSizePolicy, QSpacerItem, QMessageBox
)
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QPixmap
import cv2
from ultralytics import YOLO
from PyQt5.QtWidgets import QSlider
import pandas as pd
from PyQt5.QtWidgets import QCheckBox

from models import model_operations


class AnalyseOneImageUI(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Analyse One Image")
        self.resize(800, 600)
        self.image_path = None
        self.processed_image_path = None 
        self.confidence_level = 0.5
        self.report_results = ""
        self.init_ui()

    def init_ui(self):
        # Main horizontal layout
        layout = QHBoxLayout()

        # Left column with action buttons
        left_layout = QVBoxLayout()

        # Manual threshold checkbox
        self.manual_thresh_checkbox = QCheckBox("Manual threshold")
        self.manual_thresh_checkbox.stateChanged.connect(self.toggle_manual_threshold)
        left_layout.addWidget(self.manual_thresh_checkbox)

        # Threshold slider (hidden by default)
        self.thresh_slider = QSlider(Qt.Horizontal)
        self.thresh_slider.setMinimum(0)
        self.thresh_slider.setMaximum(255)
        self.thresh_slider.setValue(127)
        self.thresh_slider.setTickPosition(QSlider.TicksBelow)
        self.thresh_slider.setTickInterval(10)
        self.thresh_slider.valueChanged.connect(self.update_manual_threshold_preview)
        self.thresh_slider.setVisible(False)
        left_layout.addWidget(self.thresh_slider)

        self.thresh_value_label = QLabel(f"Threshold: {self.thresh_slider.value()}")
        self.thresh_value_label.setVisible(False)
        left_layout.addWidget(self.thresh_value_label)

        # Confidence slider
        conf_label_title = QLabel("Confidence level:")
        self.conf_label_value = QLabel(f"{self.confidence_level:.2f}")
        self.conf_label_value.setAlignment(Qt.AlignCenter)

        self.conf_slider = QSlider(Qt.Horizontal)
        self.conf_slider.setMinimum(0)
        self.conf_slider.setMaximum(100)
        self.conf_slider.setValue(int(self.confidence_level * 100))
        self.conf_slider.setTickPosition(QSlider.TicksBelow)
        self.conf_slider.setTickInterval(10)
        self.conf_slider.valueChanged.connect(self.update_confidence)

        left_layout.addWidget(conf_label_title)
        left_layout.addWidget(self.conf_slider)
        left_layout.addWidget(self.conf_label_value)

        self.btn_autoselect = QPushButton("Apply autoselection")
        btn_manual = QPushButton("Manual selection")
        self.btn_update_result = QPushButton("Update results")
        self.btn_report = QPushButton("Generate report")
        self.btn_all_tresh = QPushButton("Save results for all treshs")

        self.btn_autoselect.clicked.connect(self.apply_autoselection)

        self.btn_report.clicked.connect(self.save_csv_report)
        self.btn_update_result.clicked.connect(self.generate_csv_report)
        self.btn_all_tresh.clicked.connect(self.generate_result_for_each_treshold)

        for btn in [self.btn_autoselect, btn_manual,self.btn_update_result, self.btn_report,self.btn_all_tresh]:
            btn.setMinimumHeight(40)
            left_layout.addWidget(btn)

        self.report_results_label = QLabel(f"Report results: \nFirst apply autoselection")
        left_layout.addWidget(self.report_results_label)

        left_layout.addSpacerItem(QSpacerItem(20, 40, QSizePolicy.Minimum, QSizePolicy.Expanding))

        # Center area for image
        center_layout = QVBoxLayout()
        self.image_label = QLabel("No image selected")
        self.image_label.setAlignment(Qt.AlignCenter)
        self.image_label.setStyleSheet("border: 1px solid gray; background: #fafafa;")

        self.image_label.setScaledContents(True)

        center_layout.addWidget(self.image_label)

        btn_select_image = QPushButton("Select Image")
        btn_select_image.clicked.connect(self.open_image)
        left_layout.addWidget(btn_select_image, alignment=Qt.AlignCenter)

        # Save button
        self.btn_save_image = QPushButton("Save Processed Image")
        self.btn_save_image.clicked.connect(self.save_processed_image)
        self.btn_save_image.setEnabled(False)
        left_layout.addWidget(self.btn_save_image, alignment=Qt.AlignCenter)

        # Add layouts with stretch factors
        layout.addLayout(left_layout, stretch=1)   # narrow column
        layout.addLayout(center_layout, stretch=4)  # wide image area

        self.setLayout(layout)

    def update_confidence(self, value):
        self.confidence_level = value / 100.0
        self.conf_label_value.setText(f"{self.confidence_level:.2f}")

    def open_image(self):
        fname, _ = QFileDialog.getOpenFileName(self, "Select Image", "", "Images (*.png *.jpg *.jpeg *.bmp)")
        if fname:
            self.image_path = fname
            self.show_image(fname)

    def show_image(self, path):
        pixmap = QPixmap(path)
        if not pixmap.isNull():
            # Scale to fit available label size
            scaled_pixmap = pixmap.scaled(
                self.image_label.width()-2,
                self.image_label.height()-2,
                Qt.KeepAspectRatio,
                Qt.SmoothTransformation
            )
            self.image_label.setPixmap(scaled_pixmap)
        else:
            self.image_label.setText("Could not load image")

    def resizeEvent(self, event):
        """Rescale image dynamically when window is resized."""
        if self.image_label.pixmap():
            if self.processed_image_path and os.path.exists(self.processed_image_path):
                self.show_image(self.processed_image_path)
            elif self.image_path:
                self.show_image(self.image_path)
        super().resizeEvent(event)

    def apply_autoselection(self):
        if not self.image_path:
            QMessageBox.warning(self, "No image", "Please select an image first.")
            return

        save_path = os.path.join(os.path.dirname(self.image_path), "detected_image.jpg")

        # Run YOLO detection
        self.results = model_operations.get_searching_results(
            self.image_path, 
            confidence_level=self.confidence_level, 
            save_path=save_path, 
            result_preview_form="id"
            )

        # Display processed image
        self.show_image(save_path)
        self.btn_save_image.setEnabled(True)
        self.processed_image_path = save_path
        self.generate_csv_report()

    def generate_csv_report(self,optional_tresh=None):
        if not self.processed_image_path or not os.path.exists(self.processed_image_path):
            QMessageBox.warning(self, "No processed image", "Please run autoselection first.")
            return
        

        treshold = None
        if self.manual_thresh_checkbox.checkState() == Qt.Checked:
            treshold = self.thresh_slider.value()
            print(f"common_tresh: {self.thresh_slider.value()}")
        if optional_tresh != None:
            treshold = optional_tresh
        
        self.report_df = model_operations.get_results_df(self.image_path,self.results,common_tresh=treshold)

        specified_types_count_predicted,df = model_operations.full_analyse(self.report_df,proube_volume_ml=6,is_pred=True)

        self.report_results = specified_types_count_predicted[["bacteria_type", "count"]].to_string(index=False)
        self.report_results_label.setText(f"Report results:\n{self.report_results}")

    def save_csv_report(self):
        if not self.report_df:
            QMessageBox.warning(self, "No processed image", "Please run autoselection first.")
            return

        fname, _ = QFileDialog.getSaveFileName(self, "Save Processed Report", "", "Report (*.csv)")
        if fname:
            self.report_df.to_csv(f"{fname}.csv",index=False)
            QMessageBox.information(self, "Saved", f"Processed report saved to:\n{fname}.csv")
    
    def generate_result_for_each_treshold(self):
        all_results = []

        # Collect all possible bacteria types across thresholds
        all_types = set()
        for i in range(1, 255):
            self.report_df = model_operations.get_results_df(self.image_path, self.results, common_tresh=i)
            specified_types_count_predicted, df = model_operations.full_analyse(
                self.report_df, proube_volume_ml=6, is_pred=True
            )
            all_types.update(specified_types_count_predicted["bacteria_type"].tolist())
        all_types = sorted(list(all_types))

        # Collect counts for each threshold
        for i in range(1, 255):
            self.report_df = model_operations.get_results_df(self.image_path, self.results, common_tresh=i)
            specified_types_count_predicted, df = model_operations.full_analyse(
                self.report_df, proube_volume_ml=6, is_pred=True
            )

            # Ensure all bacteria types are present
            df_complete = pd.DataFrame({"bacteria_type": all_types})
            df_complete = df_complete.merge(specified_types_count_predicted, on="bacteria_type", how="left")
            df_complete["count"] = df_complete["count"].fillna(0)
            df_complete["threshold"] = i
            all_results.append(df_complete)

        # Concatenate all thresholds
        df_all = pd.concat(all_results, ignore_index=True)

        # Pivot: thresholds as rows, bacteria types as columns
        df_pivot = df_all.pivot_table(index="threshold", columns="bacteria_type", values="count", fill_value=0)

        # Optional: reset index so threshold becomes a column
        df_pivot.reset_index(inplace=True)

        df_pivot.to_csv("all_result_for_tresh1-255_pivoted.csv", index=False)



    def save_processed_image(self):
        if not self.processed_image_path or not os.path.exists(self.processed_image_path):
            QMessageBox.warning(self, "No processed image", "Please run autoselection first.")
            return

        fname, _ = QFileDialog.getSaveFileName(self, "Save Processed Image", "", "Images (*.png *.jpg *.jpeg *.bmp)")
        if fname:
            cv2.imwrite(fname, cv2.imread(self.processed_image_path))
            QMessageBox.information(self, "Saved", f"Processed image saved to:\n{fname}")

    def toggle_manual_threshold(self, state):
        # Show slider only if checkbox is checked
        manual = state == Qt.Checked
        self.thresh_slider.setVisible(manual)
        self.thresh_value_label.setVisible(manual)
        if manual and self.image_path:
            self.update_manual_threshold_preview(self.thresh_slider.value())

    def update_manual_threshold_preview(self, value):
        self.thresh_value_label.setText(f"Threshold: {value}")
        if not self.image_path:
            return
        if self.report_results:
            self.report_results = self.generate_csv_report(optional_tresh=value)

        # Generate overlay using manual threshold
        threshold_value, mask, overlay = model_operations.get_tresh_mask_for_img(self.image_path, fixed=value)
        # Save to a temporary file to display
        temp_path = os.path.join(os.path.dirname(self.image_path), "manual_thresh_preview.jpg")
        cv2.imwrite(temp_path, overlay)
        self.show_image(temp_path)



class MainUI(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("bAIcteria")
        self.resize(500, 400)
        self.init_ui()

    def init_ui(self):
        layout = QVBoxLayout()

        logo = QLabel("bAIcteria")
        logo.setAlignment(Qt.AlignCenter)
        logo.setStyleSheet("font-size: 28px; font-weight: bold;")
        layout.addWidget(logo)

        layout.addSpacerItem(QSpacerItem(20, 40, QSizePolicy.Minimum, QSizePolicy.Expanding))

        btn_analyse_one = QPushButton("Analyse one image")
        btn_analyse_dir = QPushButton("Analyse all the images in directory")
        btn_convert_report = QPushButton("Convert report .csr to full report")
        btn_settings = QPushButton("Settings")

        btn_analyse_one.clicked.connect(self.open_analyse_one)

        for btn in [btn_analyse_one, btn_analyse_dir, btn_convert_report, btn_settings]:
            btn.setMinimumHeight(40)
            layout.addWidget(btn)

        layout.addSpacerItem(QSpacerItem(20, 40, QSizePolicy.Minimum, QSizePolicy.Expanding))

        bottom_layout = QHBoxLayout()
        bottom_layout.addStretch()
        version_label = QLabel("Version 1.0.0")
        version_label.setStyleSheet("color: gray; font-size: 12px;")
        bottom_layout.addWidget(version_label)
        layout.addLayout(bottom_layout)

        self.setLayout(layout)

    def open_analyse_one(self):
        self.analyse_window = AnalyseOneImageUI()
        self.analyse_window.show()


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainUI()
    window.show()
    sys.exit(app.exec_())
