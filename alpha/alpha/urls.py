"""alpha URL Configuration

The `urlpatterns` list routes URLs to views. For more information please see:
    https://docs.djangoproject.com/en/2.2/topics/http/urls/
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
from django.contrib import admin as dj_admin
from django.urls import path, include
from django.views.generic.base import RedirectView
from django.contrib.staticfiles.views import serve
from alphatest import views as alphatest_view
from one import views as one_view
from dataManage import views as dm_view
from rest_framework import routers

urlpatterns = [
    # path('favicon.ico', serve, {'path': 'common_static/images/favicon.ico'}),

    path('test/', alphatest_view.index),
    path('trans/', one_view.trans_home, name='trans_home'),

    path('alpha/data-manage/', dm_view.alpha_data_manage, name='data_manage'),
    path('alpha/data-manage/admin/', dj_admin.site.urls),
    # path('alpha/data-manage/xm-admin', xm_admin.site.urls),
    path('alpha/data-manage/console/', dm_view.alpha_data_console, name='data_console'),
    path('alpha/data-manage/demo/', dm_view.alpha_demo, name='data_demo'),

    path('model-api/alpha-demo/save_and_run/', dm_view.AlphaDemoViewSet.save_and_run),
    path('model-api/alpha-demo/save/', dm_view.AlphaDemoViewSet.save)

]
