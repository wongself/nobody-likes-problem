from django.urls import path

from . import views

urlpatterns = [
    # page
    path('', views.page_extract, name='home'),
    path('extract/', views.page_extract, name='extract'),
    # query
    path('query_extract/', views.query_extract, name='query_extract'),
    path('query_contrast/', views.query_contrast, name='query_contrast'),
]
