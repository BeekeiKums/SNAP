from django import forms
from .models import UserAccount, Category, DataItem, Profile
from django.contrib.auth.models import User
from django.contrib.auth.forms import UserCreationForm as BaseUserCreationForm

class CategoryForm(forms.ModelForm):
    class Meta:
        model = Category
        fields = ['name', 'description', 'type']

class UserAccountForm(forms.ModelForm):
    ROLE_CHOICES = [
        ('admin', 'Admin'),
        ('businessman', 'Businessman'),
        ('data_analyst', 'Data Analyst'),
        ('content_creator', 'Content Creator'),
    ]
    role = forms.ChoiceField(choices=ROLE_CHOICES, required=True)

    class Meta:
        model = UserAccount
        fields = ['role']

class BusinessmanForm(UserAccountForm):
    class Meta:
        model = UserAccount
        fields = ['role']

class ContentCreatorForm(UserAccountForm):
    class Meta:
        model = UserAccount
        fields = ['role']

class DataAnalystForm(UserAccountForm):
    class Meta:
        model = UserAccount
        fields = ['role']

class ProfileForm(forms.ModelForm):
    class Meta:
        model = Profile
        fields = ['first_name', 'last_name', 'company', 'profile_picture']

class VisibilitySettingsForm(forms.ModelForm):
    class Meta:
        model = DataItem
        fields = ['visibility']
        widgets = {
            'visibility': forms.RadioSelect
        }

class UserCreationForm(BaseUserCreationForm):
    email = forms.EmailField(required=True)

    class Meta:
        model = User
        fields = ('username', 'email', 'password1', 'password2')








