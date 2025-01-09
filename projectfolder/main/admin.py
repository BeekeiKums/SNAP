from django.contrib import admin
from .models import Category, UserAccount, Data , SocialMediaData

# Register the models
admin.site.register(Category)
admin.site.register(Data)

@admin.register(SocialMediaData)
class SocialMediaDataAdmin(admin.ModelAdmin):
    list_display = ('platform', 'created_at')
    search_fields = ('platform',)




class UserAccountAdmin(admin.ModelAdmin):
    list_display = ('username', 'email', 'role')
    list_filter = ('role',)

admin.site.register(UserAccount, UserAccountAdmin)



