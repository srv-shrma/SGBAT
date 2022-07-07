from base64 import urlsafe_b64encode
from re import U
from django.template import loader
# from django.forms import Input
from django.utils.http import urlsafe_base64_encode, urlsafe_base64_decode
from email.message import EmailMessage
from django.shortcuts import redirect, render
from django.http import HttpResponse
from django.contrib.auth.models import User
from django.contrib import messages
from django.contrib.auth import authenticate, login, logout
from matplotlib.pyplot import title
from requests import request
import authentication
from authentication.forms import MyfileUploadForm
from gfg import settings
from django.core.mail import send_mail, EmailMessage
from django.contrib.sites.shortcuts import get_current_site
from django.template.loader import render_to_string
from django.utils.encoding import force_bytes, force_str
from .tokens import generate_token
from .models import file_upload

from .main import *
from fpdf import FPDF
from django.core.files.base import ContentFile, File
# import joblib
# import keras
# from tensorflow.python import tf2

# Create your views here.
def home(request):
    return render(request, "/GFG/templates/index.html")

def about(request):
    return render(request,'/GFG/templates/about.html')

# def adlogin(request):
#     return render(request,'/GFG/templates/authentication/adlogin.html')

# def adhome(request):
#    return render(request,'/GFG/templates/authentication/adhome.html')


def signup(request):

    if request.method == "POST":
        #username = request.POST.get('username')
        username = request.POST['username']
        fname = request.POST['fname']
        lname = request.POST['lname']
        email = request.POST['email']
        pass1 = request.POST['pass1']
        pass2 = request.POST['pass2']

        if User.objects.filter(username=username):
            messages.error(request, "patient ID already exists, choose another one.")
            return redirect('home')
        
        if User.objects.filter(email=email):
            messages.error(request, "email already registered")
            return redirect('home')
        
        if len(username)>10:
            messages.error(request, "patient ID must be under 10 characters")

        if pass1 != pass2:
            messages.error(request, "passwords didn't match.")

        if not username.isalnum():
            messages.error(request, "patient ID must be alphanumeric")
            return redirect('home')



        #here User is a model which we imported from django.contrib.auth.models
        myuser = User.objects.create_user(username, email, pass1)
        myuser.first_name = fname 
        myuser.last_name = lname 
        myuser.is_active = False
        myuser.save()

        messages.success(request, "your account has been successfully created.")

        #welcome email:

        subject= "welcome to website"
        message= "hello "+ myuser.first_name + "!\n"+ "Welcome to this website and thank you for registering!\nWe have also sent you a confirmation email. Please click the link in that email in order to activate your account. You can only login once you have activated your account./nThank you!!"
        from_email = settings.EMAIL_HOST_USER
        to_list = [myuser.email]
        send_mail(subject, message, from_email, to_list, fail_silently=True)

        #email address confirmation email:

        current_site = get_current_site(request)
        email_subject = "Confirm your email."
        message2= render_to_string('email_confirmation.html',{
        'name': myuser.first_name, 
        'domain': current_site.domain, 
        'uid': urlsafe_base64_encode(force_bytes(myuser.pk)),
        'token': generate_token.make_token(myuser),
        })
        #creating email object now:
        email = EmailMessage(
            email_subject, message2, settings.EMAIL_HOST_USER, [myuser.email],
        )
        email.fail_silently = True
        email.send()

        return redirect('sign in')
    
    
    return render(request, "signup.html")

def home(request):
    return render(request, "index.html")
   

def signin(request):

    if request.method == 'POST':
        username = request.POST['username']
        pass1 = request.POST['pass1']

        #authenticating the user:
        user = authenticate(username=username, password = pass1)

        if user is not None and user.is_active:
            login(request, user)
            fname = user.first_name
            return render(request, "index.html", {'fname':fname})

        else:
            messages.error(request, "you have entered wrong credentials, sign in again.")
            return redirect('home')

    return render(request, "signin.html")

def signout(request):
    logout(request)
    messages.success(request, "logged out successfully")
    return redirect('home')


def activate(request, uidb64, token):
    try:
        uid = force_str(urlsafe_base64_decode(uidb64)) 
        #force_text is for decoding the the special tokens and checking 
        # whether this token was given to a particular user or not
        myuser = User.objects.get(pk=uid)
    except (TypeError, ValueError, OverflowError, User.DoesNotExist):
        myuser = None
    
    
    if myuser is not None and generate_token.check_token(myuser, token):
        myuser.is_active = True
        myuser.save()
        login(request, myuser)
        return redirect('home')
    
    else:
        return render(request, 'activation_failed.html')
        

def home(request):
    return render(request,'signin.html')

def about(request):
    return render(request,'about.html')

def adlogin(request):

    if request.method == 'POST':
        username = request.POST['username']
        pass1 = request.POST['pass1']

        #authenticating the user:
        user = authenticate(username=username, password = pass1)

        if user is not None and user.is_staff:
            login(request, user)
            # fname = user.first_name
            return redirect('adhome')#, {'fname':fname})

        else:
            messages.error(request, "you have entered wrong credentials, sign in again.")
            return redirect('adlogin')

    # return render(request, "authentication/signin.html")
    else:
        # print("kkkk")
        return render(request,'adlogin.html')

def home(request):
    return render(request,'adlogin.html')

# def adhome(request):
#    return render(request,'authentication/adhome.html')



def adhome(request):
    # print("inside adhome")

    if request.method == 'POST':
        form = MyfileUploadForm(request.POST, request.FILES)


        print(form.as_p)
        
        if form.is_valid():
            name = form.cleaned_data['file_name']
            the_files_1 = form.cleaned_data['files_data1']
            the_files_2 = form.cleaned_data['files_data2']

            newfile1 = file_upload(file_name = name+"1", my_file = the_files_1)
            newfile1.save() 

            newfile2 = file_upload(file_name = name+"2", my_file = the_files_2)
            newfile2.save()

            file_path_1 = "media/"+newfile1.my_file.name
            file_path_2 = "media/"+newfile2.my_file.name

            return GENERATE_REPORT(name, file_path_1, file_path_2) 

            # file_upload(file_name=name+"1", my_file=the_files_1).save()
            # file_upload(file_name=name+"2", my_file=the_files_2).save()
            
            # return HttpResponse("file upload")
        else:
            return HttpResponse('error')

    else:
        # print("inside adhome elsse")
        context = {
            'form':MyfileUploadForm(request.POST, request.FILES)
        }      
        
        return render(request, 'adhome.html', context)
    # return render(request,'authentication/adhome.html')

def signout(request):
    logout(request)
    messages.success(request, "logged out successfully")
    return redirect('home')
        
# def compute(request):
#     f1 = file_upload.objects.get(file_name="test1").url
#     f2 = file_upload.objects.get(file_name="test2")

#     # cls = joblib.load('CNN1.sav')

#     # ans = cls.predict(f1, f2)

#     print(f1)

#     return render(request, 'authentication/adhome.html')

def show_file(request):
    # this for testing 
    search_id = request.POST.get('input_text')
    print(search_id)
    data = file_upload.objects.get(file_name=search_id)

    context = {
        'data':data 
        }

    return render(request, 'index.html', context)

# def adhome(request):
#    return render(request,'authentication/adhome.html')

def GENERATE_REPORT(user_name, file_path_1, file_path_2):
    result, acc = GET_SLEEP_STAGE(file_path_1, file_path_2)

    print("Result : {}, Accuracy : {}", result, acc)

    id = user_name[-3:]
    result = str(result)
    acc = str(acc)
    return pdfgen(user_name, id, file_path_1, file_path_2, result, acc)


def pdfgen(user, id, file_path_1, file_path_2, result, acc):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size=20)
    pdf.cell(400, 50, txt="System Generated Brain Analysis Tool", ln=1, align="L")
    pdf.set_font("Arial", size=14)
    pdf.cell(200, 10, txt="Patient Name: "+user, ln=1, align="L")
    pdf.set_font("Arial", size=14)
    pdf.cell(200, 10, txt="Patient Id: "+id, ln=1, align="L")
    pdf.set_font("Arial", size=14)
    pdf.cell(200, 10, txt="Reference File 1: "+file_path_1, ln=1, align="L")
    pdf.set_font("Arial", size=14)
    pdf.cell(200, 10, txt="Reference File 2: "+file_path_2, ln=1, align="L")
    pdf.set_font("Arial", size=14)
    pdf.cell(500, 10, txt="Sleep Disorder Stage: "+result, ln=1, align="L")
    pdf.set_font("Arial", size=14)
    pdf.cell(500, 10, txt="Prediction level: "+acc, ln=1, align="L")
    pdf.set_font("Arial", size=14)
    pdf.cell(500, 10, txt="----------------------------------", ln=1, align="L")
    pdf.set_font("Arial", size=14)
    pdf.cell(500, 10, txt=" \" Stage 0: wakefulness", ln=1, align="L")
    pdf.set_font("Arial", size=14)
    pdf.cell(500, 10, txt=" Stage 1: light sleep", ln=1, align="L")
    pdf.set_font("Arial", size=14)
    pdf.cell(500, 10, txt=" Stage 2: deeper sleep", ln=1, align="L")
    pdf.set_font("Arial", size=14)
    pdf.cell(500, 10, txt=" Stage 3: deep sleep", ln=1, align="L")
    pdf.set_font("Arial", size=14)
    pdf.cell(500, 10, txt=" Stage 4: rapid eye movement \"", ln=1, align="L")
    pdf.output('buffer/'+user+'.pdf')  
    # f = request.FILES['buffer/'+user+'.pdf']  
    with open('buffer/'+user+'.pdf','rb') as f:
        newf = file_upload(file_name = user, my_file = File(f))
        newf.save() 

    return redirect('adhome')