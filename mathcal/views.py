from django.http import JsonResponse
from django.shortcuts import render
import json
from rest_framework import generics
from rest_framework.response import Response
from .models import Chapter, Formula, CalculationResult, Question, Topic
from django.utils.decorators import method_decorator
from .serializers import ChapterSerializer, FormulaSerializer, CalculationResultSerializer
import numpy as np 
import sympy as sp
import matplotlib
matplotlib.use('Agg')  # Set the Matplotlib backend to Agg before importing pyplot
import matplotlib.pyplot as plt
from io import BytesIO
import base64
from scipy import stats
from scipy.stats import linregress
from gtts.tts import gTTS
import PyPDF4
import io


class PlotFunctionView(generics.CreateAPIView):
    def post(self, request, *args, **kwargs):
        try:
            # Get the input function from the request
            input_function = request.data.get('function')
            print("INOUT FUNCTION ", input_function)
             # Generate a range of x values
            x_values = np.linspace(-1, 1, 3)
            # Evaluate the input function for each x value
            y_values = [eval(input_function) for x in x_values]
            curve, = plt.plot(x_values,y_values) 
            xdata = curve.get_xdata()
            ydata = curve.get_ydata()
            print("Extracting data from plot....")
            print("X data points for the plot is: ", xdata)
            print("Y data points for the plot is: ", ydata)
            # # Return the coordinates and the base64-encoded image
            data = {
                'x_values': xdata.tolist(),
                'y_values': ydata.tolist(),
                # 'plot_image': plot_base64,
            }

            return JsonResponse(data)
        except Exception as e:
            return JsonResponse({'error': str(e)})


class CalculationView(generics.CreateAPIView):
    queryset = CalculationResult.objects.all()
    serializer_class = CalculationResultSerializer

    def perform_create(self, serializer):
        formula_id = self.request.data.get('formula')
        input_values = self.request.data.get('input_values')
        formula = Formula.objects.get(pk=formula_id)

        if formula.name == 'Cross Product of Matrices':
            result = self.calculate_cross_product(input_values)
            result_json = json.dumps(result)
            serializer.validated_data['result'] = result_json
        elif formula.name == 'Determinant of a Matrix':
            result = self.calculate_determinant(input_values)
            serializer.validated_data['result'] = result
        elif formula.name == 'Inverse of a Matrix':
            result = self.calculate_inverse(input_values)
            serializer.validated_data['result'] = result
        elif formula.name == 'Rank of a Matrix':
            result = self.calculate_rank(input_values)
            serializer.validated_data['result'] = result
        elif formula.name == 'Transpose of a Matrix':
            result = self.calculate_transpose(input_values)
            serializer.validated_data['result'] = result
        elif formula.name == 'Eigenvalues and Eigenvectors of a Matrix':
            result = self.calculate_eigenvalues(input_values)
            serializer.validated_data['result'] = result
        elif formula.name == 'Linear Dependence of Vectors':
            result = self.determine_linear_dependence(input_values)
            serializer.validated_data['result'] = result
        elif formula.name == 'Add Vectors':
            result = self.add_vectors(input_values)
            serializer.validated_data['result'] = result
        elif formula.name == 'Subtract Vectors':
            result = self.subtract_vectors(input_values)
            serializer.validated_data['result'] = result
        elif formula.name == 'Dot Product of Vectors':
            result = self.vector_dot_product(input_values)
            serializer.validated_data['result'] = result
        elif formula.name == 'Cross Product of Vectors':
            result = self.vector_cross_product(input_values)
            serializer.validated_data['result'] = result
        elif formula.name == 'Solve Linear Equations':
            result = self.solve_linear_equations(input_values)
            serializer.validated_data['result'] = result
        elif formula.name == 'Mean':
            result = self.calculate_mean(input_values)
            serializer.validated_data['result'] = result
        elif formula.name == 'Median':
            result = self.calculate_median(input_values)
            serializer.validated_data['result'] = result
        elif formula.name == 'Mode':
            result = self.calculate_mode(input_values)
            serializer.validated_data['result'] = result
        elif formula.name == 'Variance':
            result = self.calculate_variance(input_values)
            serializer.validated_data['result'] = result
        elif formula.name == 'Standard Deviation':
            result = self.calculate_standard_deviation(input_values)
            serializer.validated_data['result'] = result
        elif formula.name == 'Correlation Coefficient':
            result = self.calculate_correlation_coefficient(input_values)
            serializer.validated_data['result'] = result
        elif formula.name == 'Regression Analysis':
            result = self.perform_regression_analysis(input_values)
            serializer.validated_data['result'] = result
        elif formula.name == 'Solve Quadratic Equations':
            result = self.solve_quadratic_equation(input_values)
            result_json = json.dumps(result)
            serializer.validated_data['result'] = result_json

        serializer.save(formula=formula, input_values=input_values)

    def calculate_cross_product(self, input_values):
        matrix1 = np.array(input_values['matrix1'])
        matrix2 = np.array(input_values['matrix2'])
        cross_product_result = np.cross(matrix1, matrix2)
        print('Cross Product ', cross_product_result)
        return cross_product_result.tolist()
    
    def calculate_determinant(self, input_values):
        matrix = np.array(input_values['matrix'])
        matrix = matrix.astype(float)
        determinant_result = np.linalg.det(matrix)
        return determinant_result
    def calculate_inverse(self, input_values):
        matrix = np.array(input_values['matrix'])
        matrix = matrix.astype(float)
        try:
            inverse_result = np.linalg.inv(matrix)
            return inverse_result.tolist()
        except np.linalg.LinAlgError:
            # Handle the case where the matrix is singular and has no inverse
            return "Matrix is singular and has no inverse"
        
    def calculate_rank(self, input_values):
        matrix = np.array(input_values['matrix'])
        matrix = matrix.astype(float)
        rank_result = np.linalg.matrix_rank(matrix)
        rank_result = int(rank_result)
        return rank_result
    
    def calculate_transpose(self, input_values):
        matrix = np.array(input_values['matrix'])
        matrix = matrix.astype(float)
        transpose_result = matrix.T.tolist()
        return transpose_result
    def calculate_eigenvalues(self, input_values):
        matrix = np.array(input_values['matrix'])
        print("INPUT matrix ", matrix)
        matrix = matrix.astype(int)
        print("INPUT matrix after ", matrix)
        w, v = np.linalg.eig(matrix)
        data = []
        eigenvalues = np.round(w, decimals=5)
        eigenvectors = np.round(v, decimals=5)
        print("Eigen values ", w.tolist())
        print("Eigen vectors", v.tolist())
        data.append({'x_values': eigenvalues.tolist()})
        data.append({'y_values': eigenvectors.tolist()})
        return data
    def determine_linear_dependence(self, input_values):
        matrix = np.array(input_values['matrix'])
        print("INPUT matrix ", matrix)
        matrix = matrix.astype(int)
        print("INPUT matrix after ", matrix)
        _, indexes = sp.Matrix(matrix).T.rref()  # T is for transpose
        print(indexes)
        print(matrix[indexes,:])
        linear_dependence = ""
        if len(indexes) == 2:
            print("linearly independant")
            linear_dependence = "linearly independant"
        else:
            print("linearly dependant")
            linear_dependence = "linearly dependant"
        data = []
        print("Input list  ", matrix.tolist())
        data.append({'input': matrix.tolist()})
        data.append({'dependence': linear_dependence})
        return data
    def add_vectors(self, input_values):
        matrix = np.array(input_values['matrix'])
        print("INPUT matrix ", matrix)
        matrix = matrix.astype(int)
        print("INPUT matrix after ", matrix)
        arr1 = matrix.tolist()[0]
        arr2 = matrix.tolist()[1]
        
        print ("1st array : ", arr1)  
        print ("2nd array : ", arr2)  
        
        out_arr = np.add(arr1, arr2)  
        data = []
        print("Input list  ", matrix.tolist())
        data.append({'input': matrix.tolist()})
        data.append({'sum': out_arr.tolist()})
        return data
    def subtract_vectors(self, input_values):
        matrix = np.array(input_values['matrix'])
        print("INPUT matrix ", matrix)
        matrix = matrix.astype(int)
        print("INPUT matrix after ", matrix)
        arr1 = matrix.tolist()[0]
        arr2 = matrix.tolist()[1]
        
        print ("1st array : ", arr1)  
        print ("2nd array : ", arr2)  
        
        out_arr = np.subtract(arr1, arr2)  
        data = []
        print("Input list  ", out_arr.tolist())
        data.append({'input': matrix.tolist()})
        data.append({'sum': out_arr.tolist()})
        return data
    def vector_dot_product(self, input_values):
        matrix = np.array(input_values['matrix'])
        print("INPUT matrix ", matrix)
        matrix = matrix.astype(int)
        print("INPUT matrix after ", matrix)
        arr1 = matrix.tolist()[0]
        arr2 = matrix.tolist()[1]
        
        print ("1st array : ", arr1)  
        print ("2nd array : ", arr2)  
        
        out_arr = np.dot(arr1, arr2)  
        data = []
        print("Input list  ", out_arr)
        data.append({'input': matrix.tolist()})
        data.append({'sum': out_arr.tolist()})
        return data
    def vector_cross_product(self, input_values):
        matrix = np.array(input_values['matrix'])
        print("INPUT matrix ", matrix)
        matrix = matrix.astype(int)
        print("INPUT matrix after ", matrix)
        arr1 = matrix.tolist()[0]
        arr2 = matrix.tolist()[1]
        
        print ("1st array : ", arr1)  
        print ("2nd array : ", arr2)  
        
        out_arr = np.cross(arr1, arr2)  
        data = []
        print("Input list  ", out_arr.tolist())
        data.append({'input': matrix.tolist()})
        data.append({'sum': out_arr.tolist()})
        return data
    def calculate_mean(self, input_values):
        data = np.array(input_values['data'])
        return np.mean(data)
    def calculate_median(self, input_values):
        data = np.array(input_values['data'])
        return np.median(data)
    def calculate_mode(self, input_values):
        data = np.array(input_values['data'])
        mode = stats.mode(data)
        return mode.mode.tolist()
    def calculate_variance(self, input_values):
        data = np.array(input_values['data'])
        return np.var(data)
    def calculate_standard_deviation(self, input_values):
        data = np.array(input_values['data'])
        return np.std(data)
    def calculate_correlation_coefficient(self, input_values):
        data_x = np.array(input_values['data_x'])
        data_y = np.array(input_values['data_y'])
        correlation = np.corrcoef(data_x, data_y)
        return correlation[0, 1]
    def perform_regression_analysis(self, input_values):
        data_x = np.array(input_values['data_x'])
        data_y = np.array(input_values['data_y'])
        slope, intercept, r_value, p_value, std_err = linregress(data_x, data_y)
        return {
            "slope": slope,
            "intercept": intercept,
            "r_value": r_value,
            "p_value": p_value,
            "std_err": std_err
        }
    def perform_regression_analysis(self, input_values):
        data_x = np.array(input_values['data_x'])
        data_y = np.array(input_values['data_y'])
        slope, intercept, r_value, p_value, std_err = linregress(data_x, data_y)
        return {
            "slope": slope,
            "intercept": intercept,
            "r_value": r_value,
            "p_value": p_value,
            "std_err": std_err
        }
    def solve_linear_equations(self, input_values):
        print("INPUT matrix ", input_values)
        coefficients = np.array(input_values['coefficients'])
        constants = np.array(input_values['constants'])
        try:
            solution = np.linalg.solve(coefficients, constants)
            return solution.tolist()
        except np.linalg.LinAlgError:
            return "No unique solution exists."
    def solve_quadratic_equation(self, input_equation):
        x = sp.symbols('x')
        eq = sp.Eq(sp.simplify(input_equation), 0)
        print("EQ ", eq)
        # Use sympy to solve for x
        solutions = sp.solve(eq, x)
        return str(solutions)

def get_questions_by_topic(request, topic_id):
    try:
        questions = Question.objects.filter(topic__pk=topic_id)
        if questions:
            data = [{
                'question_text': question.question_text,
                'choices': [
                    question.choice_1,
                    question.choice_2,
                    question.choice_3,
                    question.choice_4
                ],
                'correct_choice': question.correct_choice
            } for question in questions]

            return JsonResponse({'questions': data})
        else:
            return JsonResponse({'message': 'No questions available for this topic'})
    except Exception as e:
        return JsonResponse({'error': str(e)})
    
def get_formulas_by_chapter(request, chapter_id):
    try:
        formulas = Formula.objects.filter(chapter__pk=chapter_id)
        print("Fors ",formulas)
        allFormulas = []
        for formula in formulas:
            print("Folmula ", formula.name, "ID ", formula.id)
            obj = {"id" : formula.id, "name": formula.name}
            allFormulas.append(obj)

        return JsonResponse({'formulas': allFormulas})
    except Exception as e:
        return JsonResponse({'error': str(e)})
    

class TextToSpeech(generics.CreateAPIView):
    def post(self, request, *args, **kwargs):
        language = 'en'
        music = ''
        print(json.loads(request.body))
        query = json.loads(request.body)
        if request.method == 'POST':
            text = query['text']
            lang = query['lang']
            pdf = request.FILES['pdf'].read() or None
            if pdf:
                pdfreader = PyPDF4.PdfFileReader(io.BytesIO(pdf))
                content = ''
                for i in range(int(pdfreader.numPages)):
                    content += pdfreader.getPage(i).extractText() + "\n"
                text = content
                myobj = gTTS(text=text, lang=lang, slow= False)
                myobj.save("static/speech.mp3")
                music = 'ok'
                context = {
                    'music': music
                }
                return JsonResponse(context)
            myobj = gTTS(text=text, lang=lang, slow= False)
            myobj.save("static/speech.mp3")
            music = 'http://127.0.0.1:8000/static/speech.mp3'
            context = {
                'music': music
            }
            return JsonResponse(context)
        else:
            pass
        context = {
            'music': music
        }
        return JsonResponse(context)
