"""
ChainRad
========

File: inference software
"""


# pylint: disable=too-many-lines
#         The working code may be under the 1000 lines limit, but we think,
#         docstring with notations is useful to anybody to understand our code
#         and the way of our thinking.


# Standard library imports
from json import load as json_load
from os.path import isfile, join
import tkinter as tk
import tkinter.filedialog as filedialog

# 3rd party imports
from PIL import Image, ImageTk
import torch

from core import META_DIR, MODEL_DIR, SoloClassifier, get_headless_models, get_simple_transformer


# Global level variables
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'


class ChainRadWindow(tk.Tk):
    """
    Provide ChainRad GUI
    ====================
    """


    # pylint: disable=too-many-instance-attributes
    #         GUI instances ususally require a lot of arguments.


    BAR_DEFAULTS = {'value_text' : 'NA',
                    'value' : 0}
    BAR_FONT = {'family' : 'Arial',
                'size' : '18'}
    BAR_KEYS = ['3_1', '3_2', '3_3', '3_4', '3_5', '3_6', '3_7', '3_8', '3_9',
                '3_10', '3_11', '3_12', '3_13', '3_14']
    BAR_LABELS = {'en' : {'3_1' : '3.01 Infiltration',
                          '3_2' : '3.02 Effusion',
                          '3_3' : '3.03 Atelectasis',
                          '3_4' : '3.04 Nodule',
                          '3_5' : '3.05 Mass',
                          '3_6' : '3.06 Pneumothorax',
                          '3_7' : '3.07 Pleural thickening',
                          '3_8' : '3.08 Consolidation',
                          '3_9' : '3.09 Emphysema',
                          '3_10' : '3.10 Fibrosis',
                          '3_11' : '3.11 Pneumonia',
                          '3_12' : '3.12 Edema',
                          '3_13' : '3.13 Hernia',
                          '3_14' : '3.14 Cardiomegalys'},
                  'hu' : {'3_1' : '3.01 Infiltráció',
                          '3_2' : '3.02 Effusion',
                          '3_3' : '3.03 Atelectasis',
                          '3_4' : '3.04 Nodule',
                          '3_5' : '3.05 Mass',
                          '3_6' : '3.06 Légmell',
                          '3_7' : '3.07 Pleural thickening',
                          '3_8' : '3.08 Consolidation',
                          '3_9' : '3.09 Emfizéma',
                          '3_10' : '3.10 Fibrózis',
                          '3_11' : '3.11 Pneumonia',
                          '3_12' : '3.12 Ödéma',
                          '3_13' : '3.13 Sérv',
                          '3_14' : '3.14 Szívmegnagyobbodás'}}
    BAR_METRICS = {'3_1' : {'label_relx' : 0.0,
                            'label_rely' : 0.24,
                            'label_justify' : 'left',
                            'label_anchor' : 'nw',
                            'value_width' : 7,
                            'value_relx' : 0.385,
                            'value_rely' : 0.24,
                            'value_anchor' : 'nw',
                            'bar_color' : 'rixel_king',
                            'bar_width' : 0.24,
                            'bar_relx' : 0.5,
                            'bar_rely' : 0.24,
                            'bar_anchor' : 'nw'},
                   '3_2' : {'label_relx' : 0.0,
                            'label_rely' : 0.285,
                            'label_justify' : 'left',
                            'label_anchor' : 'nw',
                            'value_width' : 7,
                            'value_relx' : 0.385,
                            'value_rely' : 0.285,
                            'value_anchor' : 'nw',
                            'bar_color' : 'rixel_king',
                            'bar_width' : 0.24,
                            'bar_relx' : 0.5,
                            'bar_rely' : 0.285,
                            'bar_anchor' : 'nw'},
                   '3_3' : {'label_relx' : 0.0,
                            'label_rely' : 0.33,
                            'label_justify' : 'left',
                            'label_anchor' : 'nw',
                            'value_width' : 7,
                            'value_relx' : 0.385,
                            'value_rely' : 0.33,
                            'value_anchor' : 'nw',
                            'bar_color' : 'rixel_king',
                            'bar_width' : 0.24,
                            'bar_relx' : 0.5,
                            'bar_rely' : 0.33,
                            'bar_anchor' : 'nw'},
                   '3_4' : {'label_relx' : 0.0,
                            'label_rely' : 0.375,
                            'label_justify' : 'left',
                            'label_anchor' : 'nw',
                            'value_width' : 7,
                            'value_relx' : 0.385,
                            'value_rely' : 0.375,
                            'value_anchor' : 'nw',
                            'bar_color' : 'rixel_king',
                            'bar_width' : 0.24,
                            'bar_relx' : 0.5,
                            'bar_rely' : 0.375,
                            'bar_anchor' : 'nw'},
                   '3_5' : {'label_relx' : 0.0,
                            'label_rely' : 0.42,
                            'label_justify' : 'left',
                            'label_anchor' : 'nw',
                            'value_width' : 7,
                            'value_relx' : 0.385,
                            'value_rely' : 0.42,
                            'value_anchor' : 'nw',
                            'bar_color' : 'rixel_king',
                            'bar_width' : 0.24,
                            'bar_relx' : 0.5,
                            'bar_rely' : 0.42,
                            'bar_anchor' : 'nw'},
                   '3_6' : {'label_relx' : 0.0,
                            'label_rely' : 0.465,
                            'label_justify' : 'left',
                            'label_anchor' : 'nw',
                            'value_width' : 7,
                            'value_relx' : 0.385,
                            'value_rely' : 0.465,
                            'value_anchor' : 'nw',
                            'bar_color' : 'rixel_king',
                            'bar_width' : 0.24,
                            'bar_relx' : 0.5,
                            'bar_rely' : 0.465,
                            'bar_anchor' : 'nw'},
                   '3_7' : {'label_relx' : 0.0,
                            'label_rely' : 0.51,
                            'label_justify' : 'left',
                            'label_anchor' : 'nw',
                            'value_width' : 7,
                            'value_relx' : 0.385,
                            'value_rely' : 0.51,
                            'value_anchor' : 'nw',
                            'bar_color' : 'rixel_king',
                            'bar_width' : 0.24,
                            'bar_relx' : 0.5,
                            'bar_rely' : 0.51,
                            'bar_anchor' : 'nw'},
                   '3_8' : {'label_relx' : 0.0,
                            'label_rely' : 0.555,
                            'label_justify' : 'left',
                            'label_anchor' : 'nw',
                            'value_width' : 7,
                            'value_relx' : 0.385,
                            'value_rely' : 0.555,
                            'value_anchor' : 'nw',
                            'bar_color' : 'rixel_king',
                            'bar_width' : 0.24,
                            'bar_relx' : 0.5,
                            'bar_rely' : 0.555,
                            'bar_anchor' : 'nw'},
                   '3_9' : {'label_relx' : 0.0,
                            'label_rely' : 0.6,
                            'label_justify' : 'left',
                            'label_anchor' : 'nw',
                            'value_width' : 7,
                            'value_relx' : 0.385,
                            'value_rely' : 0.6,
                            'value_anchor' : 'nw',
                            'bar_color' : 'rixel_king',
                            'bar_width' : 0.24,
                            'bar_relx' : 0.5,
                            'bar_rely' : 0.6,
                            'bar_anchor' : 'nw'},
                   '3_10' : {'label_relx' : 0.0,
                             'label_rely' : 0.645,
                             'label_justify' : 'left',
                             'label_anchor' : 'nw',
                             'value_width' : 7,
                             'value_relx' : 0.385,
                             'value_rely' : 0.645,
                             'value_anchor' : 'nw',
                             'bar_color' : 'rixel_king',
                             'bar_width' : 0.24,
                             'bar_relx' : 0.5,
                             'bar_rely' : 0.645,
                             'bar_anchor' : 'nw'},
                   '3_11' : {'label_relx' : 0.0,
                             'label_rely' : 0.69,
                             'label_justify' : 'left',
                             'label_anchor' : 'nw',
                             'value_width' : 7,
                             'value_relx' : 0.385,
                             'value_rely' : 0.69,
                             'value_anchor' : 'nw',
                             'bar_color' : 'rixel_king',
                             'bar_width' : 0.24,
                             'bar_relx' : 0.5,
                             'bar_rely' : 0.69,
                             'bar_anchor' : 'nw'},
                   '3_12' : {'label_relx' : 0.0,
                             'label_rely' : 0.735,
                             'label_justify' : 'left',
                             'label_anchor' : 'nw',
                             'value_width' : 7,
                             'value_relx' : 0.385,
                             'value_rely' : 0.735,
                             'value_anchor' : 'nw',
                             'bar_color' : 'rixel_king',
                             'bar_width' : 0.24,
                             'bar_relx' : 0.5,
                             'bar_rely' : 0.735,
                             'bar_anchor' : 'nw'},
                   '3_13' : {'label_relx' : 0.0,
                             'label_rely' : 0.78,
                             'label_justify' : 'left',
                             'label_anchor' : 'nw',
                             'value_width' : 7,
                             'value_relx' : 0.385,
                             'value_rely' : 0.78,
                             'value_anchor' : 'nw',
                             'bar_color' : 'rixel_king',
                             'bar_width' : 0.24,
                             'bar_relx' : 0.5,
                             'bar_rely' : 0.78,
                             'bar_anchor' : 'nw'},
                   '3_14' : {'label_relx' : 0.0,
                             'label_rely' : 0.825,
                             'label_justify' : 'left',
                             'label_anchor' : 'nw',
                             'value_width' : 7,
                             'value_relx' : 0.385,
                             'value_rely' : 0.825,
                             'value_anchor' : 'nw',
                             'bar_color' : 'rixel_king',
                             'bar_width' : 0.24,
                             'bar_relx' : 0.5,
                             'bar_rely' : 0.825,
                             'bar_anchor' : 'nw'},
                   'common' : {'bar_height' : 0.032,
                               'bar_borderwidth' : 0,
                               'bar_highlightthickness' : 0}}
    COLORS = {'light' : {'back_1' : '#d0d0d0',
                         'back_2' : '#c0c0c0',
                         'color' : '#000000'},
              'dark' : {'back_1' : '#404040',
                        'back_2' : '#202020',
                        'color' : '#ffffff'},
              'green' : '#28a745',
              'inactive' : '#808080',
              'red' : '#dc3545',
              'rixel_king' : '#40a0e0'}
    DISEASE_IDS = {'infiltration' : '3_1', 'effusion' : '3_2',
                   'atelectasis' : '3_3', 'nodule' : '3_4', 'mass' : '3_5',
                   'pneumothorax' : '3_6', 'pleural_thickening' : '3_7',
                   'consolidation' : '3_8', 'emphysema' : '3_9',
                   'fibrosis' : '3_10', 'pneumonia' : '3_11', 'edema' : '3_12',
                   'hernia' : '3_13', 'cardiomegaly' : '3_14'}


    def __init__(self):
        """
        Initialize the object
        =====================
        """

        # pylint: disable=too-many-statements
        #         Breaking this function to functions doesn't have too much sense.

        super().__init__()
        self.ui_mode = 'light'
        self.lang = 'en'
        self.x_image = None
        self.title('ChainRad')
        self.resizable(False, True)
        self.geometry('{}x{}+0+0'.format(self.winfo_screenwidth(),
                                         self.winfo_screenheight()))
        self.state('zoomed')
        self.navbar = tk.Frame(self, background='#c0c0c0',
                               height=self.c_height(0.1))
        self.navbar.pack(side='top', fill='both')
        self.img_read = tk.PhotoImage(file=r'./src/btn_read.png')
        self.img_read_warning = tk.PhotoImage(
                                            file=r'./src/btn_read_warning.png')
        self.btn_read = tk.Button(self.navbar, background='#c0c0c0', border=0,
                                  borderwidth=0, image=self.img_read,
                                  activebackground='#c0c0c0')
        self.btn_read.pack(side='top')
        self.btn_read.place(relx=0.02, rely=0.5, anchor='w')
        self.img_left = tk.PhotoImage(file=r'./src/btn_left.png')
        self.btn_left = tk.Button(self.navbar, background='#c0c0c0', border=0,
                                  borderwidth=0, image=self.img_left,
                                  activebackground='#c0c0c0')
        self.btn_left.pack(side='top')
        self.btn_left.place(relx=0.2, rely=0.5, anchor='c')
        self.pos_state = tk.StringVar()
        self.pos_state.set('-/-')
        self.lb_pos = tk.Label(self.navbar, textvariable=self.pos_state,
                               background='#c0c0c0', font=('Arial', '24'),
                               relief='ridge', width=11)
        self.lb_pos.pack(side='top')
        self.lb_pos.place(relx=0.3, rely=0.5, anchor='c')
        self.img_right = tk.PhotoImage(file=r'./src/btn_right.png')
        self.btn_right = tk.Button(self.navbar, background='#c0c0c0', border=0,
                                   borderwidth=0, image=self.img_right,
                                   activebackground='#c0c0c0')
        self.btn_right.pack(side='top')
        self.btn_right.place(relx=0.4, rely=0.5, anchor='c')
        self.img_hamburger = tk.PhotoImage(file=r'./src/btn_hamburger.png')
        self.btn_hamburger = tk.Button(self.navbar, background='#c0c0c0',
                                       border=0, borderwidth=0,
                                       image=self.img_hamburger,
                                       activebackground='#c0c0c0')
        self.btn_hamburger.pack(side='top')
        self.btn_hamburger.place(relx=0.98, rely=0.5, anchor='e')
        self.statusbar = tk.Frame(self, background='#c0c0c0',
                                  height=self.c_height(0.02))
        self.statusbar.pack(side='bottom', fill='both')
        self.lang_state = tk.StringVar()
        self.lang_state.set('EN')
        self.lb_lang = tk.Label(self.statusbar, textvariable=self.lang_state,
                                background='#c0c0c0', font=('Arial', '16'),
                                relief='ridge', width=4)
        self.lb_lang.pack(side='right')
        self.ui_state = tk.StringVar()
        self.ui_state.set('LM')
        self.lb_night = tk.Label(self.statusbar, textvariable=self.ui_state,
                                 background='#c0c0c0', font=('Arial', '16'),
                                 relief='ridge', width=4)
        self.lb_night.pack(side='right')
        self.status = tk.StringVar()
        self.lb_status = tk.Label(self.statusbar, textvariable=self.status,
                                  background='#c0c0c0', anchor='w',
                                  font=('Arial', '16'), justify='left',
                                  relief='ridge')
        self.lb_status.pack(side='left', fill='both', expand='yes')
        self.status.set('Waiting for user input...')
        self.main = tk.Frame(self, background='#d0d0d0',
                             height=self.c_height(0.88))
        self.main.pack(side='top', fill='both', expand='yes')
        self.p_canvas = tk.Canvas(self.main, width=self.c_width(0.48),
                                  height=self.c_height(0.79),
                                  background='#d0d0d0', highlightthickness=0)
        self.p_canvas.pack(side='left', fill='both', expand='yes')
        self.p_canvas.place(relx=0.01, rely=0.01, anchor='nw')
        self.bars_make()
        self.x_canvas = tk.Canvas(self.main, width=self.c_width(0.48),
                                  height=self.c_height(0.79),
                                  background='#d0d0d0', highlightthickness=0)
        self.x_canvas.pack(side='right', fill='both', expand='yes')
        self.x_canvas.place(relx=0.99, rely=0.01, anchor='ne')
        self.btn_read.bind('<Button-1>', self.open_files)
        self.btn_left.bind('<Button-1>', self.list_down)
        self.btn_right.bind('<Button-1>', self.list_up)
        self.lb_night.bind('<Button-1>', self.ui_switch)
        self.lb_lang.bind('<Button-1>', self.lang_switch)
        for key in self.BAR_KEYS:
            self.bar_set(key, 0)
        self.predictions = []
        self.pred_pos = 0


    def bars_make(self):
        """
        Make bars
        =========
        """

        self.bars, self.bar_labels, self.bar_values = {}, {}, {}
        for key in ChainRadWindow.BAR_KEYS:
            self.bars[key] = {}
            self.bar_labels[key] = tk.StringVar()
            self.bar_labels[key].set(ChainRadWindow.BAR_LABELS[self.lang][key])
            self.bar_values[key] = tk.StringVar()
            self.bar_values[key].set('N/A')
            self.bars[key]['value'] = tk.Label(self.p_canvas,
                                               textvariable=self
                                               .bar_values[key],
                                               background=ChainRadWindow
                                               .COLORS[self.ui_mode]['back_1'],
                                               font=(ChainRadWindow
                                                     .BAR_FONT['family'],
                                                     ChainRadWindow
                                                     .BAR_FONT['size']),
                                               relief='ridge',
                                               width=ChainRadWindow
                                               .BAR_METRICS[key]['value_width'])
            self.bars[key]['value'].pack(side='left')
            self.bars[key]['value'].place(relx=ChainRadWindow
                                          .BAR_METRICS[key]['value_relx'],
                                          rely=ChainRadWindow
                                          .BAR_METRICS[key]['value_rely'],
                                          anchor=ChainRadWindow
                                          .BAR_METRICS[key]['value_anchor'])
            self.bars[key]['label'] = tk.Label(self.p_canvas,
                                               textvariable=self
                                               .bar_labels[key],
                                               background=ChainRadWindow
                                               .COLORS[self.ui_mode]['back_1'],
                                               font=(ChainRadWindow
                                                     .BAR_FONT['family'],
                                                     ChainRadWindow
                                                     .BAR_FONT['size']),
                                               justify=ChainRadWindow
                                               .BAR_METRICS[key]
                                               ['label_justify'])
            self.bars[key]['label'].pack(side='left')
            self.bars[key]['label'].place(relx=ChainRadWindow
                                          .BAR_METRICS[key]['label_relx'],
                                          rely=ChainRadWindow
                                          .BAR_METRICS[key]['label_rely'],
                                          anchor=ChainRadWindow
                                          .BAR_METRICS[key]['label_anchor'])
            self.bars[key]['bar_nest'] = tk.Frame(self.p_canvas,
                                                  background=ChainRadWindow
                                                  .COLORS[self.ui_mode]
                                                  ['back_2'],
                                                  height=self
                  .c_height(ChainRadWindow.BAR_METRICS['common']['bar_height']),
                                                  width=self
                      .c_width(ChainRadWindow.BAR_METRICS[key]['bar_width']),
                                                  borderwidth=ChainRadWindow
                                                  .BAR_METRICS['common']
                                                  ['bar_borderwidth'],
                                                  highlightthickness=
                                                  ChainRadWindow
                                                  .BAR_METRICS['common']
                                                  ['bar_highlightthickness'])
            self.bars[key]['bar_nest'].pack(side='left', fill='both',
                                            expand='yes')
            self.bars[key]['bar_nest'].place(relx=ChainRadWindow
                                             .BAR_METRICS[key]['bar_relx'],
                                             rely=ChainRadWindow
                                             .BAR_METRICS[key]['bar_rely'],
                                             anchor=ChainRadWindow
                                             .BAR_METRICS[key]['bar_anchor'])
            self.bars[key]['bar'] = tk.Frame(self.p_canvas,
                                             background=ChainRadWindow
                                             .COLORS[ChainRadWindow
                                             .BAR_METRICS[key]['bar_color']],
                                             height=self
                 .c_height(ChainRadWindow.BAR_METRICS['common']['bar_height']),
                                             width=self
                         .c_width(ChainRadWindow.BAR_METRICS[key]['bar_width']),
                                             borderwidth=ChainRadWindow
                                             .BAR_METRICS['common']
                                             ['bar_borderwidth'],
                                             highlightthickness=ChainRadWindow
                                             .BAR_METRICS['common']
                                             ['bar_highlightthickness'])
            self.bars[key]['bar'].pack(side='left', fill='both', expand='yes')
            self.bars[key]['bar'].place(relx=ChainRadWindow
                                        .BAR_METRICS[key]['bar_relx'],
                                        rely=ChainRadWindow
                                        .BAR_METRICS[key]['bar_rely'],
                                        anchor=ChainRadWindow
                                        .BAR_METRICS[key]['bar_anchor'])


    def bar_set(self, key : str, value : float):
        """
        Set bar's state
        ===============

        Parameters
        ----------
        key : str
            ID of the bar to set.
        value : float
            Value to apply to the bar.
        """

        if value == 1:
            self.bars[key]['bar'].config(width=self.c_width(
                    ChainRadWindow.BAR_METRICS[key]['bar_width']))
            self.bar_values[key].set('YES')
        else:
            self.bars[key]['bar'].config(width=0)
            self.bar_values[key].set('NO')


    def c_height(self, rate : float =1.0) -> int:
        """
        Calculate height in pixel
        =========================

        Parameters
        ----------
        rate : float
            Rate to use in calculation.

        Returns
        -------
        int
            Pixel value.
        """

        return int(float(self.winfo_screenheight()) * float(rate))


    def c_width(self, rate : float =1.0) -> int:
        """
        Calculate width in pixel
        ========================

        Parameters
        ----------
        rate : float
            Rate to use in calculation.

        Returns
        -------
        int
            Pixel value.
        """

        return int(float(self.winfo_screenwidth()) * float(rate))


    def lang_switch(self, *args):
        """
        Switch language of the UI
        =========================

        Parameters
        ----------
        positional arguments : any
            Not used at the moment. Only added because of TKinter's
            requirements.
        """

        # pylint: disable=unused-argument
        #         Positional arguments are used to provide compatibility with
        #         TKinter.

        if self.lang == 'en':
            self.lang = 'hu'
            self.lang_state.set('HU')
        else:
            self.lang = 'en'
            self.lang_state.set('EN')
        self.ui_refresh()


    def list_down(self, *args):
        """
        Decrease list position
        ======================

        Parameters
        ----------
        positional arguments : any
            Not used at the moment. Only added because of TKinter's
            requirements.
        """

        # pylint: disable=unused-argument
        #         Positional arguments are used to provide compatibility with
        #         TKinter.

        if self.pred_pos > 0:
            self.pred_pos -= 1
            self.update_screen()


    def list_up(self, *args):
        """
        Increase list position
        ======================

        Parameters
        ----------
        positional arguments : any
            Not used at the moment. Only added because of TKinter's
            requirements.
        """

        # pylint: disable=unused-argument
        #         Positional arguments are used to provide compatibility with
        #         TKinter.

        if self.pred_pos < len(self.predictions) - 1:
            self.pred_pos += 1
            self.update_screen()


    def open_files(self, *args):
        """
        Open files
        ==========

        Parameters
        ----------
        positional arguments : any
            Not used at the moment. Only added because of TKinter's
            requirements.
        """

        # pylint: disable=unused-argument
        #         Positional arguments are used to provide compatibility with
        #         TKinter.

        filelist = filedialog.askopenfilenames()
        self.btn_read.configure(image=self.img_read_warning)
        if len(filelist) > 0:
            predictions = predict(filelist)
            self.predictions = []
            self.pred_pos = 0
            for i, prediction in enumerate(predictions):
                data_list = {}
                data_list['image'] = filelist[i]
                data_list['bars'] = {}
                for key, value in prediction.items():
                    data_list['bars'][self.DISEASE_IDS[key]] = value
                self.predictions.append(data_list)
            self.update_screen()
        self.btn_read.configure(image=self.img_read)


    def set_image(self, filename: str):
        """
        Set image on the image canvas
        =============================

        Parameters
        ----------
        filename : str
            Name of the file to read.
        """

        image = Image.open(filename)
        v_rate = self.x_canvas.winfo_height() / image.height
        h_rate = self.x_canvas.winfo_width() / image.width
        if h_rate < v_rate:
            new_width = int(image.width * h_rate)
            new_height = int(image.height * h_rate)
        else:
            new_width = int(image.width * v_rate)
            new_height = int(image.height * v_rate)
        new_x = int((self.x_canvas.winfo_width() - new_width) / 2)
        new_y = int((self.x_canvas.winfo_height() - new_height) / 2)
        self.x_image = ImageTk.PhotoImage(image.resize((new_width, new_height)))
        self.x_canvas.delete()
        self.x_canvas.create_image(new_x, new_y, image=self.x_image,
                                   anchor='nw')


    def ui_refresh(self):
        """
        Refresh the UI
        ==============
        """

        self.navbar.config(background=ChainRadWindow
                           .COLORS[self.ui_mode]['back_2'])
        self.btn_read.config(background=ChainRadWindow
                             .COLORS[self.ui_mode]['back_2'],
                             activebackground=ChainRadWindow
                             .COLORS[self.ui_mode]['back_2'])
        self.btn_left.config(background=ChainRadWindow
                             .COLORS[self.ui_mode]['back_2'],
                             activebackground=ChainRadWindow
                             .COLORS[self.ui_mode]['back_2'])
        self.lb_pos.config(background=ChainRadWindow
                           .COLORS[self.ui_mode]['back_2'],
                           foreground=ChainRadWindow
                           .COLORS[self.ui_mode]['color'])
        self.btn_right.config(background=ChainRadWindow
                              .COLORS[self.ui_mode]['back_2'],
                              activebackground=ChainRadWindow
                              .COLORS[self.ui_mode]['back_2'])
        self.btn_hamburger.config(background=ChainRadWindow
                                  .COLORS[self.ui_mode]['back_2'],
                                  activebackground=ChainRadWindow
                                  .COLORS[self.ui_mode]['back_2'])
        self.statusbar.config(background=ChainRadWindow
                              .COLORS[self.ui_mode]['back_2'])
        self.lb_lang.config(background=ChainRadWindow
                            .COLORS[self.ui_mode]['back_2'],
                            foreground=ChainRadWindow
                            .COLORS[self.ui_mode]['color'])
        self.lb_night.config(background=ChainRadWindow
                             .COLORS[self.ui_mode]['back_2'],
                             foreground=ChainRadWindow
                             .COLORS[self.ui_mode]['color'])
        self.lb_status.config(background=ChainRadWindow
                              .COLORS[self.ui_mode]['back_2'],
                              foreground=ChainRadWindow
                              .COLORS[self.ui_mode]['color'])
        self.main.config(background=ChainRadWindow
                         .COLORS[self.ui_mode]['back_1'])
        self.p_canvas.config(background=ChainRadWindow
                             .COLORS[self.ui_mode]['back_1'])
        self.x_canvas.config(background=ChainRadWindow
                             .COLORS[self.ui_mode]['back_1'])
        for key in ChainRadWindow.BAR_KEYS:
            self.bar_labels[key].set(ChainRadWindow
                                     .BAR_LABELS[self.lang][key])
            self.bars[key]['value'].config(background=ChainRadWindow
                                           .COLORS[self.ui_mode]['back_1'],
                                           foreground=ChainRadWindow
                                           .COLORS[self.ui_mode]['color'])
            self.bars[key]['label'].config(background=ChainRadWindow
                                           .COLORS[self.ui_mode]['back_1'],
                                           foreground=ChainRadWindow
                                           .COLORS[self.ui_mode]['color'])
            self.bars[key]['bar_nest'].config(background=ChainRadWindow
                                              .COLORS[self.ui_mode]['back_2'])


    def ui_switch(self, *args):
        """
        Switch UI mode
        ==============

        Parameters
        ----------
        positional arguments : any
            Not used at the moment. Only added because of TKinter's
            requirements.
        """

        # pylint: disable=unused-argument
        #         Positional arguments are used to provide compatibility with
        #         TKinter.

        if self.ui_mode == 'light':
            self.ui_mode = 'dark'
            self.ui_state.set('DM')
        else:
            self.ui_mode = 'light'
            self.ui_state.set('LM')
        self.ui_refresh()


    def update_screen(self):
        """
        Update screen
        =============
        """

        if len(self.predictions) > 0:
            self.pos_state.set('{}/{}'.format(self.pred_pos + 1,
                               len(self.predictions)))
            self.set_image(self.predictions[self.pred_pos]['image'])
            for key, value in self.predictions[self.pred_pos]['bars'].items():
                self.bar_set(key, value)
        else:
            self.pos_state.set('-/-')



class SessionSetup:
    """
    Singleton to provide session level variables
    ============================================
    """


    __diseases = {}
    __headles_models = {}
    __locked = False
    __trained_models = {}
    __transformer = lambda x: x
    __tresholds = {}


    @classmethod
    def diseases(cls) -> dict:
        """
        Get diseases
        ============

        Returns
        -------
        dict
            Dictionary of disease names.
        """

        return cls.__diseases


    @classmethod
    def headless_models(cls) -> dict:
        """
        Get headles models
        ==================

        Returns
        -------
        dcit
            Dictionary of headless models.
        """

        return cls.__headles_models


    @classmethod
    def is_configured(cls) -> bool:
        """
        Check whether the singleton class is already configured or not
        ==============================================================

        Returns
        -------
        bool
            Ture if the class is already configured, False if not.
        """

        return all([len(cls.__diseases > 0), len(cls.__headles_models > 0),
                    len(cls.__trained_models > 0), len(cls.__tresholds > 0)])


    @classmethod
    def is_locked(cls) -> bool:
        """
        Check whether the singleton class is locked or not
        ==================================================

        Returns
        -------
        bool
            Ture if the class is locked, False if not.
        """

        return cls.__locked


    @classmethod
    def keys(cls) -> list:
        """
        Get exisiting disease keys
        ==========================

        Returns
        -------
        list
            The list of existing disease keys.
        """

        return list(cls.__diseases.keys())


    @classmethod
    def lock(cls):
        """
        Lock the class
        ==============

        Raises
        ------
        PermissionError
            When the session is locked.
        """

        if cls.is_locked():
            raise PermissionError('ChainRad cannot lock a locked session.')
        cls.__locked = True


    @classmethod
    def setup(cls):
        """
        Set up session level variables
        ==============================

        Raises
        ------
        FileNotFoundError
            When the chainrad_diseases.json file doesn't exist.
        RuntimeError
            When no disease data was added to the sassion.

        See also
        --------
            PermissionError : SessionSetup.lock()
        """

        cls.lock()
        if not isfile(join(META_DIR, 'chainrad_diseases.json')):
            raise FileNotFoundError('ChainRad requires information about ' +
                                    'diseases. Inference is not available.')
        with open(join(META_DIR, 'chainrad_diseases.json'), 'r',
                  encoding='utf8') as instream:
            json_data = json_load(instream)
        for key, data in json_data.items():
            if isfile(join(MODEL_DIR, '{}.statedict'.format(key))
               ) and 'name' in data.keys() and 'treshold' in data.keys():
                cls.__trained_models[key] = SoloClassifier()
                cls.__trained_models[key].load_state_dict(torch.load(
                                join(MODEL_DIR, '{}.statedict'.format(key))))
                for param in cls.__trained_models[key].parameters():
                    param.requires_grad = False
                cls.__trained_models[key].eval()
                cls.__diseases[key] = data['name']
                cls.__tresholds[key] = data['treshold']
        if len(cls.__diseases) == 0:
            raise RuntimeError('ChainRad couldn\'t add any disease.')
        cls.__headles_models = get_headless_models()
        cls.__transformer = get_simple_transformer()
        cls.unlock()


    @classmethod
    def trained_models(cls) -> dict:
        """
        Get trained models
        ==================

        Returns
        -------
        dcit
            Dictionary of trained models.
        """

        return cls.__trained_models


    @classmethod
    def transformer(cls) -> dict:
        """
        Get transformer
        ===============

        Returns
        -------
        dcit
            Dictionary of transformer.
        """

        return cls.__transformer


    @classmethod
    def tresholds(cls) -> dict:
        """
        Get tresholds
        =============

        Returns
        -------
        dcit
            Dictionary of tresholds.
        """

        return cls.__tresholds


    @classmethod
    def unlock(cls):
        """
        Unlock the class
        ================
        """

        cls.__locked = False


def apply_treshold(raw_predicition : float, key : str) -> int:
    """
    Applies treshold on a raw predicition
    =====================================

    Parameters
    ----------
    raw_predicition : float
        The raw prediction of the model.
    key : str
        Name of the treshold to apply.

    Returns
    -------
    int
        The real predicton. 1 if disease is predicted, 0 if not.
    """

    if key in SessionSetup.tresholds().keys():
        result = 0 if raw_predicition < SessionSetup.tresholds()[key] else 1
    else:
        result = 0 if raw_predicition < 0.5 else 1
    return result


def main():
    """
    Provides main functionality
    ===========================
    """

    print('Initializing ChainRad... ', end='')
    SessionSetup.setup()
    print('Done.')
    app = ChainRadWindow()
    app.mainloop()


def predict(filelist : list) -> list:
    """
    Predict diseases from images
    ============================

    Parameters
    ----------
    filelist : list
        List of files to use as inputs.

    Returns
    -------
    list[dict]
        List of predictions. Predictions are in the form of a Dictionary where
        key is disease ID and value is 1 if the disease is predicted, 0 if not.

    Raises
    ------
    FileNotFoundError
        When a file in the filelist doesn't exist.
    """

    # pylint: disable=no-member
    #         toch has a member function cat()
    #         Link: https://pytorch.org/docs/stable/generated/torch.cat.html

    SessionSetup.lock()
    headless_outs = []
    transformer = SessionSetup.transformer()
    for headless_model in SessionSetup.headless_models().values():
        headless_model.to(DEVICE)
    for filename in filelist:
        if not isfile(filename):
            raise FileNotFoundError('Source file "{}" doesn\'t exist.'
                                    .format(filename))
        img = Image.open(filename).convert('RGB')
        img = transformer(img).unsqueeze(0).to(DEVICE)
        flats = []
        for headless_model in SessionSetup.headless_models().values():
            flats.append(headless_model(img).squeeze(0).detach().cpu())
        final = torch.cat(flats)
        headless_outs.append(final.detach().clone())
        del final
    for headless_model in SessionSetup.headless_models().values():
        headless_model.to('cpu')
    result = [{} for i in range(len(headless_outs))]
    for key in SessionSetup.keys():
        model = SessionSetup.trained_models()[key]
        model.to(DEVICE)
        for i, headles_out in enumerate(headless_outs):
            headles_out = headles_out.to(DEVICE)
            predicition = model(headles_out)
            headles_out.to('cpu')
            result[i][key] = apply_treshold(predicition, key)
        model.to('cpu')
    SessionSetup.unlock()
    return result


if __name__ == '__main__':
    main()
