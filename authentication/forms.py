from django import forms
from .models import file_upload


class MyfileUploadForm(forms.Form):
    # class Meta:
        # model = file_upload
        file_name = forms.CharField(widget=forms.TextInput(attrs={'class': 'form-control'}))
        files_data1 = forms.FileField(widget=forms.FileInput(attrs={'class':'form-control'}))
        files_data2 = forms.FileField(widget=forms.FileInput(attrs={'class':'form-control'}))

        