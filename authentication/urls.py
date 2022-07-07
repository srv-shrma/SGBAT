from django.contrib import admin
from django.urls import path, include
from showstatic import views
from . import views

urlpatterns = [
    path('admin/', admin.site.urls),
    path('',views.home, name="home"),
    path('about', views.about, name="about"),
    path('adlogin', views.adlogin, name="adlogin"),
    path('adhome', views.adhome, name="adhome"),
    # path('adhome/', views.compute, name="compute"),
    path('signup', views.signup, name="sign up"),
    path('index/file', views.show_file, name="sho"),
    path('index', views.signin, name="index"),
    path('signin', views.signin, name="sign in"),
    path('signout', views.signout, name="sign out"),
    # path('view', views.show_file, name="view"),
    path('activate/<uidb64>/<token>', views.activate, name="activate"),
]
