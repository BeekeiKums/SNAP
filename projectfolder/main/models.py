from django.db import models
from django.contrib.auth.models import User
import uuid
from django.conf import settings
from neomodel import StructuredNode, StringProperty, IntegerProperty, RelationshipTo

class Person(StructuredNode):
    name = StringProperty(unique_index=True, required=True)
    age = IntegerProperty()

    # Relationships
    friends = RelationshipTo('Person', 'FRIEND')

class Movie(StructuredNode):
    title = StringProperty(unique_index=True, required=True)
    year = IntegerProperty()


    
    
    
class Category(models.Model):
    
   
    name = models.CharField(max_length=100)
    description = models.TextField()

    def __str__(self):
        return f"{self.name}"

class UserAccount(models.Model):
    
    
    
    user_id = models.CharField(max_length=100, unique=True, default=uuid.uuid4)  # Explicitly map MongoDB _id
    username = models.CharField(max_length=100, unique=True)
    email = models.EmailField(unique=True)
    password = models.CharField(max_length=100)
    ROLE_CHOICES = [
        ('admin', 'Admin'),
        ('businessman', 'Businessman'),
        ('content_creator', 'Content Creator'),
        ('data_analyst', 'Data Analyst'),
    ]
    
    role = models.CharField(max_length=20, choices=ROLE_CHOICES, default='admin')

    def __str__(self):
        return f"{self.username} - {self.get_role_display()}"


class Businessman(models.Model):
    
    user_id = models.CharField(max_length=100, unique=True, default=uuid.uuid4)  # Explicitly map MongoDB _id
    username = models.CharField(max_length=100, unique=True)
    email = models.EmailField(unique=True)
    password = models.CharField(max_length=100)
    
    def __str__(self):
        return f"{self.username}"
    
class ContentCreator(models.Model):
    
    user_id = models.CharField(max_length = 100, unique=True, default=uuid.uuid4)
    username = models.CharField(max_length=100, unique=True)
    password = models.CharField(max_length=100)
    email = models.EmailField(unique=True) 
    
    def __str__(self):
        return f"{self.username}"
    
class DataAnalyst(models.Model):
    
    user_id = models.CharField(max_length = 100, unique=True, default=uuid.uuid4)
    username = models.CharField(max_length=100, unique=True)
    password = models.CharField(max_length=100)
    email = models.EmailField(unique=True) 
    
    def __str__(self):
        return f"{self.username}"    
    

class ExtractData(models.Model):
    platform = models.CharField(max_length=100)
    businessman = models.ForeignKey('auth.User', on_delete=models.CASCADE)
    data = models.TextField()
    extracted_at = models.DateTimeField(auto_now_add=True)
    
    
    def __str__(self):
        return f"{self.platform} data for {self.businessman.username}"
    
    

class Data(models.Model):
    name = models.CharField(max_length=100)
    age = models.IntegerField()
    from_location = models.CharField(max_length=100)
    klout_score = models.FloatField()

    def __str__(self):
        return self.name    
    


class Profile(models.Model):
    profile_id = models.CharField(max_length=100, unique=True, default=uuid.uuid4)
    first_name = models.CharField(max_length=100)
    last_name = models.CharField(max_length=100)
    company = models.CharField(max_length=100)
    timezone = models.CharField(max_length=100)
    
    def __str__(self):
        return f"{self.first_name} {self.last_name}"
    
    

class DataItem(models.Model):
    VISIBILITY_CHOICES = [
        ('private', 'Private'),
        ('restricted', 'Restricted'),
        ('public', 'Public'),
    ]

    businessman = models.ForeignKey(UserAccount, on_delete=models.CASCADE)
    name = models.CharField(max_length=255)
    description = models.TextField(blank=True, null=True)
    visibility = models.CharField(max_length=20, choices=VISIBILITY_CHOICES, default='private')
    created_at = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return self.name


class Testimonial(models.Model):
    user = models.ForeignKey(UserAccount, on_delete= models.CASCADE)
    content = models.TextField()
    rating = models.PositiveIntegerField()  # 1 to 5 stars
    created_at = models.DateTimeField(auto_now_add=True)
    
    def __str__(self):
        return f"{self.user.username}'s Testimonial"


