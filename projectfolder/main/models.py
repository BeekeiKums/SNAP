from django.db import models
from django.contrib.auth.models import User
import uuid
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
    role = models.CharField(max_length=20, choices=ROLE_CHOICES, default='businessman')

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
    

class SocialMediaData(models.Model):
    platform = models.CharField(max_length=50)
    data = models.TextField()
    created_at = models.DateTimeField(auto_now_add=True)
    
    
    def __str__ (self):
        return f"{self.platform} Data on {self.created_at}"
    



