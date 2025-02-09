"""myproject URL Configuration

The `urlpatterns` list routes URLs to views. For more information please see:
    https://docs.djangoproject.com/en/4.1/topics/http/urls/
Examples:
Function views
    1. Add an import:  from my_app import views
    2. Add a URL to urlpatterns:  path('', views.home, name='home')
Class-based views
    1. Add an import:  from other_app.views import Home
    2. Add a URL to urlpatterns:  path('', Home.as_view(), name='home')
Including another URLconf
    1. Import the include() function: from django.urls import include, path
    2. Add a URL to urlpatterns:  path('blog/', include('blog.urls'))
"""
from django.contrib import admin
from django.urls import path, include
from django.shortcuts import redirect
from main import views  # Import views from the main app

urlpatterns = [
    path('', lambda request: redirect('marketing_page')),  # Redirect root to marketing_page
    path('', include('main.urls')),  # Include all urls from the main app
    path('login/', views.login, name='login'),  # Ensure this points to the correct login view
    path('preds/', views.preds, name='preds'),
    path('predict_engagement/', views.predict_engagement, name='predict_engagement'),
    path('upload-csv/', views.upload_csv, name='upload_csv'),  # Handles CSV uploads
    path('graph_view/', views.graph_view, name='graph_view'),  # Ensure graph_view is defined
]

from django.conf import settings
from django.conf.urls.static import static

if settings.DEBUG:
    urlpatterns += static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)

# USE NGINX OR APACHE TO SERVE MEDIA FILES IN PRODUCTION