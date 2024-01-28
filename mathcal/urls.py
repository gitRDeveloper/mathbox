from django.urls import path
from .views import *
from . import views
urlpatterns = [
    # path('chapters/', ChapterList.as_view(), name='chapter-list'),
    # path('formulas/', FormulaList.as_view(), name='formula-list'),
    # path('results/', CalculationResultList.as_view(), name='result-list'),
    path('calculate/', CalculationView.as_view(), name='calculate'),
    path('plot_function/', PlotFunctionView.as_view(), name='plot_function'),
    path('get_questions/<int:topic_id>/', views.get_questions_by_topic, name='get_questions_by_topic'),
    path('get_formulas_by_chapter/<int:chapter_id>/', views.get_formulas_by_chapter, name='get_formulas_by_chapter'),
    path('TextToSpeech/', TextToSpeech.as_view(), name='TextToSpeech'),
]