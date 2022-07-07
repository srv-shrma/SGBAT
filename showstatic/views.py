from django.shortcuts import render

def home(request):
   return render(request,'index.html')


def about(request):
   return render(request,'about.html')

def adlogin(request):
   return render(request,'adlogin.html')

def adhome(request):
   return render(request,'adhome.html')
