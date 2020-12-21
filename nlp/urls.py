from django.urls import path

from . import views

urlpatterns = [
    # page
    path('', views.page_extract, name='home'),
    path('extract/', views.page_extract, name='extract'),
    path('dialog/', views.page_dialog, name='dialog'),
    path('template/', views.page_template, name='template'),  # 复制该行，粘贴在该行之上，并将 template 字段进行重命名，就像 page_extract 一样。

    # query
    path('query_contrast/', views.query_contrast, name='query_contrast'),
    path('query_extract/', views.query_extract, name='query_extract'),
    path('query_dialog/', views.query_dialog, name='query_dialog'),
    path('query_template/', views.query_template, name='query_template'),  # 复制该行，粘贴在该行之上，并将 template 字段进行重命名，就像 query_extract 一样。
]
