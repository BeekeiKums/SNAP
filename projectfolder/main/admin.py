from django.contrib import admin
from .models import Category, UserAccount, Data 

# Register the models
admin.site.register(Category)
admin.site.register(Data)
admin.site.register(UserAccount)


class UserAccountAdmin(admin.ModelAdmin):
    list_display = ('username', 'email', 'role')
    list_filter = ('role',)





