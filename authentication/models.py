# from django.contrib.auth.models import AbstractUser

from django.db import models
# from django.forms import ModelForm

class file_upload(models.Model):

    ids = models.AutoField(primary_key=True)
    file_name = models.CharField(max_length=255)
    my_file = models.FileField(upload_to='')
    # my_file2 = models.FileField(upload_to='',  blank=True)

    def __str__(self):
        return self.file_name

# class User(AbstractUser):
#     is_admin= models.BooleanField('Is admin', default=False)
#     is_customer = models.BooleanField('Is customer', default=False)
#     is_employee = models.BooleanField('Is employee', default=False)
    
    
