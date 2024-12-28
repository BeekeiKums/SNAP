from django import forms
from .models import UserAccount, Category , Data

class CategoryForm(forms.ModelForm):
    class Meta:
        model = Category
        fields = ['name', 'description']
               
class UserAccountForm(forms.ModelForm):
    class Meta:
        model = UserAccount
        fields = ['username', 'email', 'password', 'role']
        widgets = {
            'password': forms.PasswordInput(attrs={'placeholder': 'Enter Password'}),
       }
        
        
class BusinessmanForm(UserAccountForm):
    class Meta:
        model = UserAccount
        fields = ['username', 'email', 'password']
        widgets = {
            'password': forms.PasswordInput(attrs={'placeholder': 'Enter Password'}),
       }
        
class ContentCreatorForm(UserAccountForm):
    
    class Meta:
        model = UserAccount
        fields = ['username', 'email', 'password']   
        widgets = {
            'password ': forms.PasswordInput(attrs={'placeholder': 'Enter Password'}),
        }     
       
class DataAnalystForm(UserAccountForm):
    
    class Meta:
        model = UserAccount
        fields = ['username', 'email', 'password']   
        widgets = {
            'password ': forms.PasswordInput(attrs={'placeholder': 'Enter Password'}),
        }       

            
 
class DataForm(forms.ModelForm):
    class Meta:
        model = Data
        fields = ['name', 'age', 'from_location', 'klout_score']  