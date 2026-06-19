# -*- coding: utf-8 -*-

import os
import sys
import re
import shutil
from glob import glob
import pandas as pd


# Legacy version from genpage.py
class gen_page_legacy:

    def __init__(self, data_path):
        '''
        This function takes in the path to the data folder and returns the HTML files for the OPLS-DA plots.
        Parameters
        ----------
        data_path: str
            The path to the data folder.
        gen_page_legacy(data_path).get_files()
        '''
        self.data_path = data_path

        if data_path[-1] == '/':
            #remove the last /
            data_path = data_path[:-1]

        else:
            data_path = data_path



        #check data_path is a string
        if not isinstance(data_path, str):
            raise ValueError("data_path should be a string")

        #check data_path is a directory
        if not os.path.isdir(data_path):
            raise ValueError("data_path should be a directory")

        #check if data_path is empty
        if not os.listdir(data_path):
            raise ValueError("data_path should not be empty")

        #check if data_path contains the necessary files
        if not os.path.exists(data_path+'/element/hist_plot'):
            raise ValueError("data_path should contain a folder named 'element' with a folder named 'hist_plot'")
        if not os.path.exists(data_path+'/element/Lingress'):
            raise ValueError("data_path should contain a folder named 'element' with a folder named 'Lingress'")
        if not os.path.exists(data_path+'/element/loading_plot'):
            raise ValueError("data_path should contain a folder named 'element' with a folder named 'loading_plot'")
        if not os.path.exists(data_path+'/element/s_plot'):
            raise ValueError("data_path should contain a folder named 'element' with a folder named 's_plot'")
        if not os.path.exists(data_path+'/element/score_plot'):
            raise ValueError("data_path should contain a folder named 'element' with a folder named 'score_plot'")
        if not os.path.exists(data_path+'/element/VIP_scores'):
            raise ValueError("data_path should contain a folder named 'element' with a folder named 'VIP_scores'")

        #change directory to data_path
        os.chdir(data_path)

    def get_files(self):

        data_path = self.data_path


        hist_plot = glob(pathname= data_path+'/element/hist_plot/*.html')
        Lingress_ = glob(pathname= data_path+'/element/Lingress/*.html')
        loading_plot = glob(pathname= data_path+'/element/loading_plot/*.html')
        s_plot = glob(pathname= data_path+'/element/s_plot/*.html')
        score_plot = glob(pathname= data_path+'/element/score_plot/*.html')
        VIP_scores = glob(pathname= data_path+'/element/VIP_scores/*.html')

        files = pd.DataFrame({'hist_plot': hist_plot, 'Lingress': Lingress_, 'loading_plot': loading_plot, 's_plot': s_plot, 'score_plot': score_plot, 'VIP_scores': VIP_scores})

        # Get the name of the files
        files['names'] = files['hist_plot'].str.split('/').str[-1].str.split('.').str[0]
        files['names'] = files['names'].str.replace('Permutation_scores_','')
        files['names'] = files['names'].str.replace(' ','_')
        #replace value in dataframe with ..
        files = files.replace(to_replace=data_path, value='..', regex=True)

        html_content_list = []

        # Iterate through the name and file_ lists to create HTML files
        for i in range(len(files)):
            html_content = f"""

        <!DOCTYPE html>
        <html lang="en">
        <head>
        <title>OPLS-DA</title>
        <meta charset="utf-8">
        <meta name="viewport" content="width=device-width, initial-scale=1">
        <meta http-equiv="X-UA-Compatible" content="IE=edge">
        <meta name="description" content="HTML5 website template">
        <meta name="keywords" content="CASIX, template, html, sass, jquery">
        <meta name="author" content="CASIX">

        <link rel="ix-icon" type="image/png" href="assets/img/logo.PNG">
        <link rel="shortcut icon" type="image/png" href="assets/img/logo.PNG">
        </head>
        <body>

        <body>

            <div class="container">
                <iframe src="{files['score_plot'][i]}"
                frameborder="0"
                width="100%"
                height="1000"></iframe>
            </div>

            <div class="container">
                <iframe src="{files['loading_plot'][i]}"
                frameborder="0"
                width="100%"
                height="1000"></iframe>
            </div>

            <div class="container">
                <iframe src="{files['s_plot'][i]}"
                frameborder="0"
                width="100%"
                height="1000"></iframe>
            </div>

            <div class="container">
                <iframe src={files['VIP_scores'][i]}"
                frameborder="0"
                width="100%"
                height="1000"></iframe>
            </div>

            <div class="container">
                <iframe src="{files['hist_plot'][i]}"
                frameborder="0"
                width="100%"
                height="1000"></iframe>

            </div>

                <div class="container">
                <iframe src="{files['Lingress'][i]}"
                frameborder="0"
                width="100%"
                height="1000"></iframe>
            </div>


        </body>
        </html>


            """
            html_content_list.append(html_content)

        for i in range(len(files)):
            # Write the HTML content to the file
            file_path = f"./main/oplsda_{files['names'][i]}.html"
            with open(file_path, "w") as html_file:
                html_file.write(html_content_list[i])

        print('HTML files created')


class gen_page:

    def __init__(self, data_path):
        '''
        This function takes in the path to the data folder and returns the HTML files for the OPLS-DA plots.
        Parameters
        ----------
        data_path: str
            The path to the data folder.
        gen_page(data_path).get_files()
        '''
        if not isinstance(data_path, str):
            raise ValueError("data_path should be a string")

        data_path = data_path.rstrip('/')
        self.data_path = data_path

        #check data_path is a directory
        if not os.path.isdir(data_path):
            raise ValueError("data_path should be a directory")

        #check if data_path is empty
        if not os.listdir(data_path):
            raise ValueError("data_path should not be empty")

        #check if data_path contains the necessary files
        if not os.path.exists(data_path+'/element/hist_plot'):
            raise ValueError("data_path should contain a folder named 'element' with a folder named 'hist_plot'")
        if not os.path.exists(data_path+'/element/Lingress'):
            raise ValueError("data_path should contain a folder named 'element' with a folder named 'Lingress'")
        if not os.path.exists(data_path+'/element/loading_plot'):
            raise ValueError("data_path should contain a folder named 'element' with a folder named 'loading_plot'")
        if not os.path.exists(data_path+'/element/s_plot'):
            raise ValueError("data_path should contain a folder named 'element' with a folder named 's_plot'")
        if not os.path.exists(data_path+'/element/score_plot'):
            raise ValueError("data_path should contain a folder named 'element' with a folder named 'score_plot'")
        if not os.path.exists(data_path+'/element/VIP_score'):
            raise ValueError("data_path should contain a folder named 'element' with a folder named 'VIP_score'")



        #change directory to data_path
        os.chdir(data_path)

    def get_files(self):

        data_path = self.data_path


        hist_plot = glob(pathname= data_path+'/element/hist_plot/*.html')
        Lingress_ = glob(pathname= data_path+'/element/Lingress/*.html')
        loading_plot = glob(pathname= data_path+'/element/loading_plot/*.html')
        s_plot = glob(pathname= data_path+'/element/s_plot/*.html')
        score_plot = glob(pathname= data_path+'/element/score_plot/*.html')
        VIP_score = glob(pathname= data_path+'/element/VIP_score/*.html')

        files = pd.DataFrame({'hist_plot': hist_plot, 'Lingress': Lingress_, 'loading_plot': loading_plot, 's_plot': s_plot,'VIP': VIP_score, 'score_plot': score_plot})

        # Get the name of the files
        files['names'] = files['hist_plot'].str.split('/').str[-1].str.split('.').str[0]
        files['names'] = files['names'].str.replace('Permutation_scores_','')
        files['names'] = files['names'].str.replace(' ','_')
        #replace value in dataframe with ..
        files = files.replace(to_replace=data_path, value='..', regex=True)

        html_content_list = []

        # Iterate through the name and file_ lists to create HTML files
        for i in range(len(files)):
            html_content = f"""

        <!DOCTYPE html>
        <html lang="en">
        <head>
        <title>OPLS-DA</title>
        <meta charset="utf-8">
        <meta name="viewport" content="width=device-width, initial-scale=1">
        <meta http-equiv="X-UA-Compatible" content="IE=edge">
        <meta name="description" content="HTML5 website template">
        <meta name="keywords" content="CASIX, template, html, sass, jquery">
        <meta name="author" content="CASIX">

        <link rel="ix-icon" type="image/png" href="assets/img/logo.PNG">
        <link rel="shortcut icon" type="image/png" href="assets/img/logo.PNG">
        </head>
        <body>

        <body>

            <div class="container">
                <iframe src="{files['score_plot'][i]}"
                frameborder="0"
                width="100%"
                height="1000"></iframe>
            </div>

            <div class="container">
                <iframe src="{files['loading_plot'][i]}"
                frameborder="0"
                width="100%"
                height="1000"></iframe>
            </div>

            <div class="container">
                <iframe src="{files['s_plot'][i]}"
                frameborder="0"
                width="100%"
                height="600"></iframe>
            </div>

            <div class="container">
                <iframe src="{files['hist_plot'][i]}"
                frameborder="0"
                width="100%" height="600"></iframe>
            </div>


            <div class="container">
                <iframe src="{files['VIP'][i]}"
                frameborder="0"
                width="100%" height="600"></iframe>
            </div>

            <div class="container">
                <iframe src="{files['Lingress'][i]}"
                frameborder="0"
                width="100%" height="600"></iframe>
            </div>


        </body>
        </html>


            """
            html_content_list.append(html_content)

        for i in range(len(files)):
            # Write the HTML content to the file
            file_path = f"./main/oplsda_{files['names'][i]}.html"
            with open(file_path, "w") as html_file:
                html_file.write(html_content_list[i])

        print('HTML files created')


class oplsda_path:

    def __init__(self, data_path):

        '''
        This function takes in the path to the data folder and returns the path to the OPLS-DA plots.
        Parameters
        ----------
        data_path: str
            The path to the data folder.
        oplsda_path(data_path).get_path()
        '''


        #check data_path is a string
        if not isinstance(data_path, str):
            raise ValueError("data_path should be a string")

        #check data_path is a directory
        if not os.path.isdir(data_path):
            raise ValueError("data_path should be a directory")

        #check if data_path is empty
        if not os.listdir(data_path):
            raise ValueError("data_path should not be empty")


        #Implement \ to / for windows
        data_path = data_path.replace('\\', '/').rstrip('/')

        self.data_path = data_path



    def make_path(self):
        """Create report folders under the provided data path and cache paths.

        Call this before `get_path()` to ensure directories exist.
        """
        data_path = self.data_path

        # Ensure directory structure exists
        os.makedirs(f"{data_path}/OPLS_DA_report", exist_ok=True)
        os.makedirs(f"{data_path}/OPLS_DA_report/main", exist_ok=True)
        os.makedirs(f"{data_path}/OPLS_DA_report/element", exist_ok=True)
        os.makedirs(f"{data_path}/OPLS_DA_report/element/hist_plot", exist_ok=True)
        os.makedirs(f"{data_path}/OPLS_DA_report/element/Lingress", exist_ok=True)
        os.makedirs(f"{data_path}/OPLS_DA_report/element/loading_plot", exist_ok=True)
        os.makedirs(f"{data_path}/OPLS_DA_report/element/s_plot", exist_ok=True)
        os.makedirs(f"{data_path}/OPLS_DA_report/element/score_plot", exist_ok=True)
        os.makedirs(f"{data_path}/OPLS_DA_report/element/VIP_score", exist_ok=True)

        # Cache path mapping
        path = {
            'main': f"{data_path}/OPLS_DA_report/main",
            'element': f"{data_path}/OPLS_DA_report/element",
            'hist_plot': f"{data_path}/OPLS_DA_report/element/hist_plot",
            'Lingress': f"{data_path}/OPLS_DA_report/element/Lingress",
            'loading_plot': f"{data_path}/OPLS_DA_report/element/loading_plot",
            's_plot': f"{data_path}/OPLS_DA_report/element/s_plot",
            'score_plot': f"{data_path}/OPLS_DA_report/element/score_plot",
            'VIP_score': f"{data_path}/OPLS_DA_report/element/VIP_score",
        }

        self.path = path
        return path

    def get_path(self):
        path = self.path
        return path
