from django.urls import path
from . import views
urlpatterns =[
    path('', views.home, name='home'),
    path('classify',views.classify_xray, name='classify_xray'),
    # path('test', views.test_, name='test')
]