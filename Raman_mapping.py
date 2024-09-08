import sys
import ZWC_Setuptool as setuptool
modules_to_check = ["sys", "numpy", "matplotlib", "PyQt6"]
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.widgets import Cursor
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from PyQt6.QtWidgets import *
from PyQt6.QtGui import *
from matplotlib.patches import Rectangle
from matplotlib.widgets import *
from PyQt6.QtCore import Qt
from lmfit import *
from lmfit.models import *
from multiprocessing import Pool, cpu_count


def Raman_singlepeak_fit(params_tuple):
    df_fitparam, data_x, data_y = params_tuple
    center_param = df_fitparam[df_fitparam["parameter"] == "center"]
    sigma_param = df_fitparam[df_fitparam["parameter"] == "sigma"]
    amplitude_param = df_fitparam[df_fitparam["parameter"] == "amplitude"]
    background_min, background_max = 800, 1000
    model = ConstantModel()
    p1 = LorentzianModel(prefix="p1_")
    params = model.make_params()
    params["c"].set(100, min=background_min, max=background_max)
    params.update(
        p1.make_params(
            center=dict(
                value=center_param["initial_value"].values[0],
                min=center_param["min"].values[0],
                max=center_param["max"].values[0],
            ),
            sigma=dict(
                value=sigma_param["initial_value"].values[0],
                min=sigma_param["min"].values[0],
                max=sigma_param["max"].values[0],
            ),
            amplitude=dict(
                value=amplitude_param["initial_value"].values[0],
                min=amplitude_param["min"].values[0],
                max=amplitude_param["max"].values[0],
            ),
        )
    )
    model = model + p1
    init = model.eval(params, x=data_x)
    result = model.fit(data_y, params, x=data_x)
    comps = result.eval_components()

    # 计算 R-squared 值
    residuals = data_y - result.best_fit
    ss_res = np.sum(residuals**2)
    ss_tot = np.sum((data_y - np.mean(data_y)) ** 2)
    r_squared = result.rsquared

    result_dict = {
        "params": result.params.valuesdict(),
        "fit_values": result.best_fit,
        "comps": {key: val for key, val in comps.items()},
        "r_squared": r_squared,
    }
    return result_dict


def parallel_fit_spectra(df_fitparam, x, maprawdata, threshold_min, threshold_max):
    params_list = [
        (
            df_fitparam,
            x[abs(x - threshold_min).argmin() : abs(x - threshold_max).argmin()],
            maprawdata[
                abs(x - threshold_min).argmin() : abs(x - threshold_max).argmin(), i
            ],
        )
        for i in range(maprawdata.shape[1])
    ]
    with Pool(cpu_count()) as pool:
        results = pool.map(Raman_singlepeak_fit, params_list)
    return np.array(results)


"PyQt6 GUI Class"


class MyWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.initUI()

    def initUI(self):
        self.tabs = QTabWidget()
        self.setCentralWidget(self.tabs)

        self.tab1 = QWidget()
        self.tabs.addTab(self.tab1, "Tab 1")

        ## main layout: [left/mid/right]
        self.tab1_layout = QHBoxLayout()
        self.tab1.setLayout(self.tab1_layout)

        ## left layout: [col1,col2,col5]
        self.left_layout = QVBoxLayout()
        self.tab1_layout.addLayout(self.left_layout)
        ## layout_col1: upload file
        self.layout_col1 = QVBoxLayout()
        self.left_layout.addLayout(self.layout_col1)
        self.button_xaxis = QPushButton("Upload x-axis")
        self.button_xaxis.clicked.connect(self.upload_xaxis)
        self.layout_col1.addWidget(self.button_xaxis)
        self.xaxis_inputpath = QLabel("")
        self.layout_col1.addWidget(self.xaxis_inputpath)
        self.button_mappingrawdata = QPushButton("Upload map raw data")
        self.button_mappingrawdata.clicked.connect(self.upload_maprawdata)
        self.layout_col1.addWidget(self.button_mappingrawdata)
        self.maprawdata_inputpath = QLabel("")
        self.layout_col1.addWidget(self.maprawdata_inputpath)
        ## layout_col2: preview spectrum
        self.layout_col2 = QVBoxLayout()
        self.left_layout.addLayout(self.layout_col2)
        self.spectrumpreview_figure, self.spectrumpreview_ax = plt.subplots()
        self.spectrumpreview_cancvas = FigureCanvas(self.spectrumpreview_figure)
        self.layout_col2.addWidget(self.spectrumpreview_cancvas)
        self.mouseposition = QLabel("")
        self.layout_col2.addWidget(self.mouseposition)
        ## Layout_col5: spectrum scale
        self.layout_col5 = QHBoxLayout()
        self.left_layout.addLayout(self.layout_col5)
        self.zoom_in = QPushButton("Zoom in")
        self.zoom_in.clicked.connect(self.previewspectrum_zoomin)
        self.layout_col5.addWidget(self.zoom_in)
        self.zoom_out = QPushButton("Zoom out")
        self.zoom_out.clicked.connect(self.previewspectrum_zoomout)
        self.layout_col5.addWidget(self.zoom_out)
        ## Layout_col7: save spectrum
        self.layout_col7 = QHBoxLayout()
        self.left_layout.addLayout(self.layout_col7)
        self.save_previewspectrum_button = QPushButton("Save spectrum")
        self.save_previewspectrum_button.clicked.connect(self.save_previewspectrum)
        self.layout_col7.addWidget(self.save_previewspectrum_button)
        self.save_previewspectrum_label = QLabel("")
        self.layout_col7.addWidget(self.save_previewspectrum_label)

        ## mid layout: [col3,col4]
        self.mid_layout = QVBoxLayout()
        self.tab1_layout.addLayout(self.mid_layout)
        ## layout_col3: intensity summation map
        self.layout_col3 = QVBoxLayout()
        self.mid_layout.addLayout(self.layout_col3)
        self.map_figure, self.map_ax = plt.subplots()
        self.map_cancvas = FigureCanvas(self.map_figure)
        self.layout_col3.addWidget(self.map_cancvas)
        self.explore_status = QLabel("Press A and move mouse to explore spectrum")
        self.layout_col3.addWidget(self.explore_status)
        ## Layout_col8: save map
        self.layout_col8 = QHBoxLayout()
        self.mid_layout.addLayout(self.layout_col8)
        self.save_intensitysummap_button = QPushButton("Save intensity summation map")
        self.save_intensitysummap_button.clicked.connect(self.save_intensitysummap)
        self.layout_col8.addWidget(self.save_intensitysummap_button)
        self.save_intensitysummap_label = QLabel("")
        self.layout_col8.addWidget(self.save_intensitysummap_label)

        self.tab2 = QWidget()
        self.tabs.addTab(self.tab2, "Tab 2")

        self.tab2_layout = QHBoxLayout()
        self.tab2.setLayout(self.tab2_layout)
        ## right layout: [col6]
        self.right_layout = QVBoxLayout()
        self.tab2_layout.addLayout(self.right_layout)
        ## layout_col6: fitting peak map
        self.layout_col6 = QVBoxLayout()
        self.right_layout.addLayout(self.layout_col6)
        ## Fitting set up button
        self.layout_col9 = QHBoxLayout()
        self.right_layout.addLayout(self.layout_col9)
        self.generateparametertable_button = QPushButton(
            "Generate Raman single peak initial guess table"
        )
        self.generateparametertable_button.clicked.connect(
            self.Ramansinglepeak_generateinitialguess
        )
        self.layout_col9.addWidget(self.generateparametertable_button)
        self.Fit_Ramansinglepeak_button = QPushButton("Fit peak preview")
        self.Fit_Ramansinglepeak_button.clicked.connect(self.Raman_singlepeak_preview)
        self.layout_col9.addWidget(self.Fit_Ramansinglepeak_button)
        ## Fitting spectrum & map
        self.layout_col10 = QVBoxLayout()
        self.right_layout.addLayout(self.layout_col10)
        self.fitpreviewspectrum_figure, self.fitpreviewspectrum_ax = plt.subplots()
        self.fitpreviewspectrum_canvas = FigureCanvas(self.fitpreviewspectrum_figure)
        self.fitpeakmap_button = QPushButton("Fitting map")
        self.fitpeakmap_button.clicked.connect(self.peak_position_fittingmap)
        self.layout_col10.addWidget(self.fitpeakmap_button)
        self.layout_col10.addWidget(self.fitpreviewspectrum_canvas)
        self.layout_col11 = QHBoxLayout()
        self.right_layout.addLayout(self.layout_col11)
        self.fitwidth_figure, self.fitwidth_ax = plt.subplots()
        self.fitwidth_canvas = FigureCanvas(self.fitwidth_figure)
        self.layout_col11.addWidget(self.fitwidth_canvas)
        self.fitpeakmap_figure, self.fitpeakmap_ax = plt.subplots()
        self.fitpeakmap_canvas = FigureCanvas(self.fitpeakmap_figure)
        self.layout_col11.addWidget(self.fitpeakmap_canvas)
        self.fitrsquare_figure, self.fitrsquare_ax = plt.subplots()
        self.fitrsquare_canvas = FigureCanvas(self.fitrsquare_figure)
        self.layout_col11.addWidget(self.fitrsquare_canvas)
        self.fitampmap_figure, self.fitampmap_ax = plt.subplots()
        self.fitampmap_canvas = FigureCanvas(self.fitampmap_figure)
        self.layout_col11.addWidget(self.fitampmap_canvas)

        self.layout_col12 = QHBoxLayout()
        self.right_layout.addLayout(self.layout_col12)

        self.savefitwidthmap_button = QPushButton("Save Fit-fwhm")
        self.savefitwidthmap_button.clicked.connect(self.savemap_Fitfwhm)
        self.layout_col12.addWidget(self.savefitwidthmap_button)

        self.savefitpeakmap_button = QPushButton("Save Fit-peak map")
        self.savefitpeakmap_button.clicked.connect(self.savemap_Fitpeak)
        self.layout_col12.addWidget(self.savefitpeakmap_button)

        self.savefitrsquaremap_button = QPushButton("Save Fit-Rsquare map")
        self.savefitrsquaremap_button.clicked.connect(self.savemap_Fitrsquare)
        self.layout_col12.addWidget(self.savefitrsquaremap_button)

        self.savefitampmap_button = QPushButton("Save Fit-amp map")
        self.savefitampmap_button.clicked.connect(self.savemap_Fitamp)
        self.layout_col12.addWidget(self.savefitampmap_button)

        # Tab3:profile analysis
        self.tab3 = QWidget()
        self.tabs.addTab(self.tab3, "Profile")

        self.tab3_layout = QHBoxLayout()
        self.tab3.setLayout(self.tab3_layout)

        self.layout_col20 = QVBoxLayout()
        self.tab3_layout.addLayout(self.layout_col20)
        self.profile_map_figure, self.profile_map_ax = plt.subplots()
        self.profile_map_canvas = FigureCanvas(self.profile_map_figure)
        self.layout_col20.addWidget(self.profile_map_canvas)

        self.layout_col21 = QVBoxLayout()
        self.tab3_layout.addLayout(self.layout_col21)
        # 右侧 - profile图
        self.profile_spectrum_figure, self.profile_spectrum_ax = plt.subplots()
        self.profile_spectrum_canvas = FigureCanvas(self.profile_spectrum_figure)
        self.layout_col21.addWidget(self.profile_spectrum_canvas)

        self.profile_figure, self.profile_ax = plt.subplots()
        self.profile_canvas = FigureCanvas(self.profile_figure)
        self.layout_col21.addWidget(self.profile_canvas)
        self.saveprofilespectrum_buttton = QPushButton("Save profile spectrum")
        self.layout_col21.addWidget(self.saveprofilespectrum_buttton)
        self.saveprofilespectrum_buttton.clicked.connect(self.saveprofile_spectrum)

        self.setGeometry(50, 50, 900, 600)
        self.setWindowTitle("Raman Mapping GUI")
        icon = QIcon("PL Mapping_PyQt6 python\icon_mapping.png")
        self.setWindowIcon(icon)
        self.show()

        # common variables
        self.maprawdata, self.xaxis = None, None
        self.pix_map = None
        self.previewspectrum_thresholdmin, self.previewspectrum_thresholdmax = (
            None,
            None,
        )
        self.minvline = self.spectrumpreview_ax.vlines(
            0, 0, 0, color="black", linestyles="dashed"
        )
        self.maxvline = self.spectrumpreview_ax.vlines(
            0, 0, 0, color="black", linestyles="dashed"
        )
        self.spectrum_preview = None
        self.original_xlim, self.original_ylim = None, None
        self.fitpreviewspectrum = None
        self.A_key_pressed = False
        self.intensitymap, self.pix_map = None, None
        self.row, self.col = None, None
        self.fitpreview_x, self.fitpreview_y = None, None
        self.fit_results = None
        self.rsquare_map, self.peakposition_map, self.amp_map, self.width_map = (
            None,
            None,
            None,
            None,
        )
        self.profile_line = None
        self.profile_points = None

    def saveprofile_spectrum(self):
        file_path, _ = QFileDialog.getSaveFileName(
            None, "Save txt File", "", "txt Files (*.txt);;All Files (*)"
        )
        if file_path:
            if not file_path.endswith(".txt"):
                file_path += "_"

        for i, (xi, yi) in enumerate(self.profile_points):
            save_path = file_path + f"({i},{xi},{yi})" + ".txt"
            self.profile_spectrum = self.maprawdata[:, yi * self.pix_map + xi]
            np.savetxt(save_path, self.profile_spectrum)
            save_path = file_path
        return

    def profile_on_press(self, event):
        if event.button == 1 and event.inaxes == self.profile_map_ax:
            self.x0, self.y0 = event.xdata, event.ydata
            if self.profile_line is not None:
                self.profile_line.remove()
                self.profile_line = None
            (self.profile_line,) = self.profile_map_ax.plot(
                [self.x0, self.x0], [self.y0, self.y0], "r-"
            )
            self.profile_map_canvas.draw()
            self.profile_spectrum_ax.cla()
            self.profile_ax.cla()
            self.profile_points = []  # 初始化 profile 点列表
        return

    def profile_on_motion(self, event):
        if (
            event.button == 1
            and self.profile_line is not None
            and event.inaxes == self.profile_map_ax
        ):
            self.x1, self.y1 = event.xdata, event.ydata
            self.profile_spectrum_ax.cla()
            self.profile_line.set_data([self.x0, self.x1], [self.y0, self.y1])
            self.update_profile()
            self.profile_spectrum_ax.plot(
                self.xaxis,
                self.maprawdata[:, int(event.ydata) * self.pix_map + int(event.xdata)],
            )
            self.profile_map_canvas.draw()
            self.profile_spectrum_canvas.draw()
            self.profile_canvas.draw()
        return

    def profile_on_release(self, event):
        if (
            event.button == 1
            and self.profile_line is not None
            and event.inaxes == self.profile_map_ax
        ):
            self.x1, self.y1 = event.xdata, event.ydata
            self.profile_line.set_data([self.x0, self.x1], [self.y0, self.y1])
            self.profile_spectrum_ax.plot(
                self.xaxis,
                self.maprawdata[:, int(event.ydata) * self.pix_map + int(event.xdata)],
            )
            self.update_profile()
            self.profile_canvas.draw()
            self.profile_spectrum_canvas.draw()
            self.profile_map_canvas.draw()
            self.x0, self.y0 = None, None
        return

    def update_profile(self):
        num_points = max(abs(int(self.x1 - self.x0)), abs(int(self.y1 - self.y0))) + 1
        num_points = max(2, num_points)

        x = np.linspace(self.x0, self.x1, num_points)
        y = np.linspace(self.y0, self.y1, num_points)

        x_indices = np.clip(np.round(x).astype(int), 0, self.intensitymap.shape[1] - 1)
        y_indices = np.clip(np.round(y).astype(int), 0, self.intensitymap.shape[0] - 1)

        profile_data = self.intensitymap[y_indices, x_indices]
        self.profile_points = list(zip(x_indices, y_indices))

        distances = np.sqrt((x - self.x0) ** 2 + (y - self.y0) ** 2)

        self.profile_ax.cla()
        self.profile_ax.plot(distances, profile_data)
        self.profile_ax.set_xlabel("Distance (pixels)")
        self.profile_ax.set_ylabel("Intensity")
        self.profile_ax.set_title("Intensity Profile")
        if len(self.profile_points) > 1:
            for i, (xi, yi) in enumerate(self.profile_points):
                if (
                    i % max(1, num_points // 20) == 0
                    or i == len(self.profile_points) - 1
                ):
                    self.profile_ax.annotate(
                        f"({i},{xi},{yi})",
                        (distances[i], profile_data[i]),
                        textcoords="offset points",
                        xytext=(0, 20),
                        ha="center",
                    )
                    profilespectrum_preview = self.maprawdata[
                        :, self.row * self.pix_map + self.col
                    ]

        if hasattr(self, "profile_points_plot"):
            self.profile_points_plot.remove()
        self.profile_points_plot = self.profile_map_ax.scatter(
            x_indices, y_indices, color="red", s=5
        )

        self.profile_canvas.draw()
        self.profile_map_canvas.draw()
        return

    def savemap_Fitfwhm(self):

        file_path, _ = QFileDialog.getSaveFileName(
            None, "Save txt File", "", "txt Files (*.txt);;All Files (*)"
        )
        if file_path:
            if not file_path.endswith(".txt"):
                file_path += ".txt"
        np.savetxt(file_path, self.width_map)
        return

    def savemap_Fitpeak(self):

        file_path, _ = QFileDialog.getSaveFileName(
            None, "Save txt File", "", "txt Files (*.txt);;All Files (*)"
        )
        if file_path:
            if not file_path.endswith(".txt"):
                file_path += ".txt"
        np.savetxt(file_path, self.peakposition_map)
        return

    def savemap_Fitamp(self):

        file_path, _ = QFileDialog.getSaveFileName(
            None, "Save txt File", "", "txt Files (*.txt);;All Files (*)"
        )
        if file_path:
            if not file_path.endswith(".txt"):
                file_path += ".txt"
        np.savetxt(file_path, self.amp_map)
        return

    def savemap_Fitrsquare(self):

        file_path, _ = QFileDialog.getSaveFileName(
            None, "Save txt File", "", "txt Files (*.txt);;All Files (*)"
        )
        if file_path:
            if not file_path.endswith(".txt"):
                file_path += ".txt"
        np.savetxt(file_path, self.rsquare_map)
        return

    def Ramansinglepeak_generateinitialguess(self):
        fit_params = {
            "parameter": ["center", "sigma", "amplitude"],
            "initial_value": [0, 0, 0],
            "min": [0, 0, 0],
            "max": [0, 0, 0],
        }
        df_fitparam = pd.DataFrame(fit_params)
        df_spectrum = pd.DataFrame(
            {"fit_datax": self.fitpreview_x, "fit_datay": self.fitpreview_y}
        )
        file_path, _ = QFileDialog.getSaveFileName(
            None, "Save Excel File", "", "Excel Files (*.xlsx);;All Files (*)"
        )
        if file_path:
            if not file_path.endswith(".xlsx"):
                file_path += ".xlsx"
            with pd.ExcelWriter(file_path) as writer:
                df_fitparam.to_excel(writer, sheet_name="Fit Parameters", index=False)
                df_spectrum.to_excel(writer, sheet_name="Spectrum Data", index=False)
        return

    def Raman_singlepeak_preview(self):
        fitparam_dialog = QFileDialog()
        fitparam_dialog.setFileMode(QFileDialog.FileMode.ExistingFile)
        fitparam_dialog.setNameFilter("fit_param (*.xlsx)")
        if fitparam_dialog.exec():
            fitparam_filepath = fitparam_dialog.selectedFiles()
            df_fitparam = pd.read_excel(fitparam_filepath[0])
            result_dict = Raman_singlepeak_fit(
                (df_fitparam, self.fitpreview_x, self.fitpreview_y)
            )
            if self.fitpreviewspectrum is not None:
                self.fitpreviewspectrum.pop(0).remove()
            self.fitpreviewspectrum_ax.plot(
                self.fitpreview_x, self.fitpreview_y, label="data"
            )
            self.fitpreviewspectrum_ax.plot(
                self.fitpreview_x, result_dict["fit_values"], "--", label="best"
            )
            self.fitpreviewspectrum_ax.legend()
            self.fitpreviewspectrum_canvas.draw()
        return

    def peak_position_fittingmap(self):
        fitparam_dialog = QFileDialog()
        fitparam_dialog.setFileMode(QFileDialog.FileMode.ExistingFile)
        fitparam_dialog.setNameFilter("fit_param (*.xlsx)")
        if fitparam_dialog.exec():
            fitparam_filepath = fitparam_dialog.selectedFiles()
            df_fitparam = pd.read_excel(fitparam_filepath[0])
            self.fit_results = parallel_fit_spectra(
                df_fitparam,
                self.xaxis,
                self.maprawdata,
                self.previewspectrum_thresholdmin,
                self.previewspectrum_thresholdmax,
            )
            self.rsquare_map = np.zeros(self.pix_map * self.pix_map)
            self.peakposition_map = np.zeros(self.pix_map * self.pix_map)
            self.amp_map = np.zeros(self.pix_map * self.pix_map)
            self.width_map = np.zeros(self.pix_map * self.pix_map)

            for i, result in enumerate(self.fit_results):
                mapdata = self.maprawdata[:, i]
                if result["r_squared"] <= 0.2:
                    self.peakposition_map[i] = np.nan
                    self.amp_map[i] = np.nan
                    self.width_map[i] = np.nan
                else:
                    self.rsquare_map[i] = result["r_squared"]
                    self.peakposition_map[i] = result["params"]["p1_center"]
                    self.amp_map[i] = result["params"]["p1_amplitude"]
                    self.width_map[i] = result["params"]["p1_fwhm"]

            self.rsquare_map = self.rsquare_map.reshape(self.pix_map, self.pix_map)

            self.peakposition_map = self.peakposition_map.reshape(
                self.pix_map, self.pix_map
            )
            self.amp_map = self.amp_map.reshape(self.pix_map, self.pix_map)
            self.width_map = self.width_map.reshape(self.pix_map, self.pix_map)

            cmap = plt.cm.viridis
            cmap.set_bad(color="black")  # 将 NaN 值显示为黑色

            fitwidthmap_cax = self.fitwidth_ax.imshow(self.width_map, cmap=cmap)
            self.fitwidth_ax.set_title("FWHM")
            self.fitwidth_ax.figure.colorbar(
                fitwidthmap_cax, ax=self.fitwidth_ax, orientation="vertical"
            )
            self.fitwidth_canvas.draw()

            fitpeakmap_cax = self.fitpeakmap_ax.imshow(self.peakposition_map, cmap=cmap)
            self.fitpeakmap_ax.set_title("Peak position")
            self.fitpeakmap_ax.figure.colorbar(
                fitpeakmap_cax, ax=self.fitpeakmap_ax, orientation="vertical"
            )
            self.fitpeakmap_canvas.draw()

            fitrsquare_cax = self.fitrsquare_ax.imshow(self.rsquare_map, cmap=cmap)
            self.fitrsquare_ax.set_title("Rsquare")
            self.fitrsquare_ax.figure.colorbar(
                fitrsquare_cax, ax=self.fitrsquare_ax, orientation="vertical"
            )

            fitamp_cax = self.fitampmap_ax.imshow(self.amp_map, cmap=cmap)
            self.fitampmap_ax.set_title("Amplitude")
            self.fitampmap_ax.figure.colorbar(
                fitamp_cax, ax=self.fitampmap_ax, orientation="vertical"
            )
            self.fitampmap_canvas.draw()

        return

    def save_intensitysummap(self):
        file_path, _ = QFileDialog.getSaveFileName(
            None, "Save Txt File", "", "Txt Files (*.txt);;All Files (*)"
        )
        if file_path:
            if not file_path.endswith(".txt"):
                file_path += "_"
        np.savetxt(
            file_path,
            self.intensitymap,
        )
        self.save_intensitysummap_label.setText(
            f"Saved intensity summation map, threshold from (min {self.previewspectrum_thresholdmin:.2f}, max {self.previewspectrum_thresholdmax:.2f})"
        )
        return

    def save_previewspectrum(self):
        self.spe_combined_array = np.column_stack((self.xaxis, self.spectrum_preview))
        file_path, _ = QFileDialog.getSaveFileName(
            None, "Save Txt File", "", "Txt Files (*.txt);;All Files (*)"
        )
        if file_path:
            if not file_path.endswith(".txt"):
                file_path += "_"
        np.savetxt(
            file_path,
            self.spe_combined_array,
        )
        self.save_previewspectrum_label.setText(
            f"Saved spectrum at (Row {self.row:.2f}, Col {self.col:.2f})"
        )
        return

    def keyPressEvent(self, event):
        if event.key() == Qt.Key.Key_A:
            self.A_key_pressed = True
            self.explore_status.setText("Exploring...")

    def keyReleaseEvent(self, event):
        if event.key() == Qt.Key.Key_A:
            self.A_key_pressed = False
            self.explore_status.setText("Press A and move mouse to explore spectrum")

    def previewspectrum_zoomout(self):
        self.spectrumpreview_ax.set_xlim(self.original_xlim)
        self.spectrumpreview_ax.set_ylim(self.original_ylim)
        self.spectrumpreview_cancvas.draw()
        return

    def previewspectrum_zoomin(self):
        def zoomin_threshold(eclick, erelease):
            x1, y1 = eclick.xdata, eclick.ydata
            x2, y2 = erelease.xdata, erelease.ydata
            self.spectrumpreview_ax.set_xlim(x1, x2)
            self.spectrumpreview_ax.set_ylim(y1, y2)
            self.spectrumpreview_cancvas.draw()
            self.previewspectrum_zoomin.set_active(False)
            return

        self.previewspectrum_zoomin = RectangleSelector(
            self.spectrumpreview_ax,
            zoomin_threshold,
            useblit=True,
            button=[1, 3],
            minspanx=5,
            minspany=5,
            spancoords="pixels",
            props=dict(alpha=0.5, facecolor="tab:blue"),
        )
        return

    def upload_xaxis(self):
        xaxis_dialog = QFileDialog()
        xaxis_dialog.setFileMode(QFileDialog.FileMode.ExistingFile)
        xaxis_dialog.setNameFilter("xaxis (*.txt)")
        if xaxis_dialog.exec():
            xaxisdata_filepath = xaxis_dialog.selectedFiles()
            if xaxisdata_filepath:
                selected_file_path = xaxisdata_filepath[0]
                self.xaxis_inputpath.setText(
                    f"Input map rawdata from {selected_file_path}"
                )
                self.xaxis = np.loadtxt(selected_file_path)

    def upload_maprawdata(self):
        maprawdata_dialog = QFileDialog()
        maprawdata_dialog.setFileMode(QFileDialog.FileMode.ExistingFile)
        maprawdata_dialog.setNameFilter("mappingrawdata (*.txt)")
        if maprawdata_dialog.exec():
            maprawdata_filepath = maprawdata_dialog.selectedFiles()
            if maprawdata_filepath:
                selected_file_path = maprawdata_filepath[0]
                self.maprawdata_inputpath.setText(
                    f"Input map rawdata from {selected_file_path}"
                )
                self.maprawdata = np.loadtxt(selected_file_path)
                self.pix_map = int((np.shape(self.maprawdata)[1]) ** 0.5)
                self.intensitymap = np.zeros([self.pix_map, self.pix_map])
                self.col, self.row = 0, 0
                rowdata_index = self.row * self.pix_map + self.col
                self.spectrum_preview = self.maprawdata[:, 0]

                self.spectrumpreview_ax.plot(
                    self.xaxis,
                    self.spectrum_preview,
                    label=f"Row {self.row}, Col {self.col}",
                )
                self.spectrumpreview_ax.legend()
                self.original_xlim = self.spectrumpreview_ax.get_xlim()
                self.original_ylim = self.spectrumpreview_ax.get_ylim()

                def mouse_move(event):
                    if self.spectrumpreview_ax == event.inaxes:
                        x, y = event.xdata, event.ydata
                        self.mouseposition.setText(
                            f"local Position: ({x:.2f}, {y:.2f})"
                        )
                    return

                def threshold(x_min, x_max):
                    for annotation in self.spectrumpreview_ax.texts:
                        annotation.remove()
                    self.minvline.remove()
                    self.maxvline.remove()
                    (
                        self.previewspectrum_thresholdmin,
                        self.previewspectrum_thresholdmax,
                    ) = (x_min, x_max)
                    self.minvline = self.spectrumpreview_ax.vlines(
                        self.previewspectrum_thresholdmin,
                        min(self.spectrum_preview),
                        max(self.spectrum_preview),
                        color="black",
                        linestyles="dashed",
                    )
                    self.maxvline = self.spectrumpreview_ax.vlines(
                        self.previewspectrum_thresholdmax,
                        min(self.spectrum_preview),
                        max(self.spectrum_preview),
                        color="black",
                        linestyles="dashed",
                    )

                    for i in range(self.pix_map):
                        for j in range(self.pix_map):
                            rowdata_index = i * self.pix_map + j
                            spe_intensity = self.maprawdata[:, rowdata_index]
                            self.intensitymap[i, j] = sum(
                                spe_intensity[
                                    abs(self.xaxis - self.previewspectrum_thresholdmin)
                                    .argmin() : abs(
                                        self.xaxis - self.previewspectrum_thresholdmax
                                    )
                                    .argmin()
                                ]
                            )
                    self.map_ax.imshow(self.intensitymap)
                    self.map_ax.set_title(
                        "Threshold range from"
                        + "({0:.2f})".format(self.previewspectrum_thresholdmin)
                        + "~"
                        + "({0:.2f})".format(self.previewspectrum_thresholdmax)
                    )
                    self.map_cancvas.draw()
                    self.spectrumpreview_cancvas.draw()

                    self.profile_map_ax.imshow(self.intensitymap)
                    self.profile_map_ax.set_title("Profile analysis map")
                    self.profile_map_canvas.draw()

                    self.previewspectrum_threshold.set_active(False)
                    return (
                        self.previewspectrum_thresholdmin,
                        self.previewspectrum_thresholdmax,
                    )

                def map_mouse_move(event):
                    if self.A_key_pressed and event.inaxes == self.map_ax:
                        self.col, self.row = int(event.xdata + 0.5), int(
                            event.ydata + 0.5
                        )
                        self.spectrumpreview_ax.cla()
                        self.spectrum_preview = self.maprawdata[
                            :, self.row * self.pix_map + self.col
                        ]
                        self.spectrumpreview_ax.plot(
                            self.xaxis,
                            self.spectrum_preview,
                            label=f"Row {self.row}, Col {self.col}",
                        )
                        self.spectrumpreview_ax.legend()
                        self.spectrumpreview_cancvas.draw()

                    if self.A_key_pressed and event.inaxes == self.fitwidth_ax:
                        self.col, self.row = int(event.xdata + 0.5), int(
                            event.ydata + 0.5
                        )
                        self.fitpreviewspectrum_ax.cla()
                        self.fitpreview_y = self.maprawdata[
                            :, self.row * self.pix_map + self.col
                        ]
                        self.fitpreview_y = self.fitpreview_y[
                            abs(self.xaxis - self.previewspectrum_thresholdmin)
                            .argmin() : abs(
                                self.xaxis - self.previewspectrum_thresholdmax
                            )
                            .argmin()
                        ]
                        self.fitpreviewspectrum_ax.plot(
                            self.fitpreview_x,
                            self.fitpreview_y,
                            label=f"Row {self.row}, Col {self.col}",
                        )
                        self.fitpreviewspectrum_ax.plot(
                            self.fitpreview_x,
                            self.fit_results[self.row * self.pix_map + self.col][
                                "fit_values"
                            ],
                            label=f"Row {self.row}, Col {self.col}, Fwhm {self.width_map[self.row][self.col]:.2f}",
                        )
                        print(
                            self.fit_results[self.row * self.pix_map + self.col][
                                "r_squared"
                            ]
                        )
                        self.fitpreviewspectrum_ax.legend(bbox_to_anchor=(1, 1))
                        self.fitpreviewspectrum_canvas.draw()

                    if self.A_key_pressed and event.inaxes == self.fitampmap_ax:
                        self.col, self.row = int(event.xdata + 0.5), int(
                            event.ydata + 0.5
                        )
                        self.fitpreviewspectrum_ax.cla()
                        self.fitpreview_y = self.maprawdata[
                            :, self.row * self.pix_map + self.col
                        ]
                        self.fitpreview_y = self.fitpreview_y[
                            abs(self.xaxis - self.previewspectrum_thresholdmin)
                            .argmin() : abs(
                                self.xaxis - self.previewspectrum_thresholdmax
                            )
                            .argmin()
                        ]
                        self.fitpreviewspectrum_ax.plot(
                            self.fitpreview_x,
                            self.fitpreview_y,
                            label=f"Row {self.row}, Col {self.col}",
                        )
                        self.fitpreviewspectrum_ax.plot(
                            self.fitpreview_x,
                            self.fit_results[self.row * self.pix_map + self.col][
                                "fit_values"
                            ],
                            label=f"Row {self.row}, Col {self.col}, Amp {self.amp_map[self.row][self.col]:.2f}",
                        )
                        print(
                            self.fit_results[self.row * self.pix_map + self.col][
                                "r_squared"
                            ]
                        )
                        self.fitpreviewspectrum_ax.legend(bbox_to_anchor=(1, 1))
                        self.fitpreviewspectrum_canvas.draw()

                    if self.A_key_pressed and event.inaxes == self.fitrsquare_ax:
                        self.col, self.row = int(event.xdata + 0.5), int(
                            event.ydata + 0.5
                        )
                        self.fitpreviewspectrum_ax.cla()
                        self.fitpreview_y = self.maprawdata[
                            :, self.row * self.pix_map + self.col
                        ]
                        self.fitpreview_y = self.fitpreview_y[
                            abs(self.xaxis - self.previewspectrum_thresholdmin)
                            .argmin() : abs(
                                self.xaxis - self.previewspectrum_thresholdmax
                            )
                            .argmin()
                        ]
                        self.fitpreviewspectrum_ax.plot(
                            self.fitpreview_x,
                            self.fitpreview_y,
                            label=f"Row {self.row}, Col {self.col}",
                        )
                        self.fitpreviewspectrum_ax.plot(
                            self.fitpreview_x,
                            self.fit_results[self.row * self.pix_map + self.col][
                                "fit_values"
                            ],
                            label=f"Row {self.row}, Col {self.col}, Rsquare {self.rsquare_map[self.row][self.col]:.2f}",
                        )
                        print(
                            self.fit_results[self.row * self.pix_map + self.col][
                                "r_squared"
                            ]
                        )
                        self.fitpreviewspectrum_ax.legend(bbox_to_anchor=(1, 1))
                        self.fitpreviewspectrum_canvas.draw()

                    if self.A_key_pressed and event.inaxes == self.fitpeakmap_ax:
                        self.col, self.row = int(event.xdata + 0.5), int(
                            event.ydata + 0.5
                        )
                        self.fitpreviewspectrum_ax.cla()
                        self.fitpreview_y = self.maprawdata[
                            :, self.row * self.pix_map + self.col
                        ]
                        self.fitpreview_y = self.fitpreview_y[
                            abs(self.xaxis - self.previewspectrum_thresholdmin)
                            .argmin() : abs(
                                self.xaxis - self.previewspectrum_thresholdmax
                            )
                            .argmin()
                        ]
                        self.fitpreviewspectrum_ax.plot(
                            self.fitpreview_x,
                            self.fitpreview_y,
                            label=f"Row {self.row}, Col {self.col}",
                        )
                        self.fitpreviewspectrum_ax.plot(
                            self.fitpreview_x,
                            self.fit_results[self.row * self.pix_map + self.col][
                                "fit_values"
                            ],
                            label=f"Row {self.row}, Col {self.col}, Peak {self.peakposition_map[self.row][self.col]:.2f}",
                        )
                        print(
                            self.fit_results[self.row * self.pix_map + self.col][
                                "r_squared"
                            ]
                        )
                        self.fitpreviewspectrum_ax.legend(bbox_to_anchor=(1, 1))
                        self.fitpreviewspectrum_canvas.draw()

                def previewspectrum_rightclick(event):
                    if event.button == 3:
                        if event.inaxes == self.spectrumpreview_ax:
                            context_menu = QMenu(self)
                            action1 = context_menu.addAction(
                                "intensity image threshold"
                            )
                            action2 = context_menu.addAction("Fit peak preview")
                            action = context_menu.exec(self.mapToGlobal(QCursor.pos()))
                            if action == action1:
                                self.previewspectrum_threshold = SpanSelector(
                                    self.spectrumpreview_ax,
                                    threshold,
                                    "horizontal",
                                    useblit=True,
                                )
                            elif action == action2:
                                self.fitpreviewspectrum_ax.cla()
                                self.fitpreviewspectrum_ax.set_title(
                                    "Fit preview spectrum"
                                )
                                self.fitpreview_x = self.xaxis[
                                    abs(self.xaxis - self.previewspectrum_thresholdmin)
                                    .argmin() : abs(
                                        self.xaxis - self.previewspectrum_thresholdmax
                                    )
                                    .argmin()
                                ]
                                self.fitpreview_y = self.spectrum_preview[
                                    abs(self.xaxis - self.previewspectrum_thresholdmin)
                                    .argmin() : abs(
                                        self.xaxis - self.previewspectrum_thresholdmax
                                    )
                                    .argmin()
                                ]
                                self.fitpreviewspectrum = (
                                    self.fitpreviewspectrum_ax.plot(
                                        self.fitpreview_x, self.fitpreview_y
                                    )
                                )
                                self.fitpreviewspectrum_canvas.draw()

                self.map_cancvas.mpl_connect("motion_notify_event", map_mouse_move)
                self.fitwidth_canvas.mpl_connect("motion_notify_event", map_mouse_move)
                self.fitrsquare_canvas.mpl_connect(
                    "motion_notify_event", map_mouse_move
                )
                self.fitampmap_canvas.mpl_connect("motion_notify_event", map_mouse_move)
                self.fitpeakmap_canvas.mpl_connect(
                    "motion_notify_event", map_mouse_move
                )
                self.spectrumpreview_cancvas.mpl_connect(
                    "motion_notify_event", mouse_move
                )
                self.spectrumpreview_cancvas.mpl_connect(
                    "button_press_event", previewspectrum_rightclick
                )
                self.profile_map_canvas.mpl_connect(
                    "button_press_event", self.profile_on_press
                )
                self.profile_map_canvas.mpl_connect(
                    "motion_notify_event", self.profile_on_motion
                )
                self.profile_map_canvas.mpl_connect(
                    "button_release_event", self.profile_on_release
                )


def main():
    app = QApplication(sys.argv)
    window = MyWindow()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
