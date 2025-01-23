from django.db import models
from django.contrib.auth.models import User
import uuid
from django.conf import settings
from neomodel import StructuredNode, StringProperty, IntegerProperty, RelationshipTo
from djongo import models
from bson import ObjectId
from django.db.models.signals import post_save
from django.dispatch import receiver
import logging
import traceback

logger = logging.getLogger(__name__)

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
    type = models.CharField(max_length=50, choices=[
        ('content_creator', 'Content Creator'),
        ('business', 'Business'),
        ('analyst', 'Analyst'),
        ('admin', 'Admin')  # Add 'admin' to choices
    ], default='admin')  # Set default value to 'admin'

    def __str__(self):
        return f"{self.name}"

class UserAccount(models.Model):
    user = models.OneToOneField(User, on_delete=models.CASCADE, default=1)  # Ensure default value
    username = models.CharField(max_length=150, unique=True)  # Add username field
    role = models.CharField(max_length=255)
    email = models.EmailField()

    def __str__(self):
        return f"{self.user.username} - {self.role}"

@receiver(post_save, sender=User)
def create_user_account(sender, instance, created, **kwargs):
    if created:
        try:
            if not UserAccount.objects.filter(user=instance).exists():
                UserAccount.objects.create(user=instance, username=instance.username)
                logger.info(f"UserAccount created for user: {instance.username}")
            else:
                logger.warning(f"UserAccount already exists for user: {instance.username}")
        except Exception as e:
            logger.error(f"Error creating UserAccount for user: {instance.username} - {e}")
            logger.error(traceback.format_exc())

@receiver(post_save, sender=User)
def save_user_account(sender, instance, **kwargs):
    try:
        if hasattr(instance, 'useraccount'):
            instance.useraccount.save()
            logger.info(f"UserAccount saved for user: {instance.username}")
        else:
            logger.warning(f"UserAccount does not exist for user: {instance.username}")
    except Exception as e:
        logger.error(f"Error saving UserAccount for user: {instance.username} - {e}")
        logger.error(traceback.format_exc())

class Businessman(models.Model):
    user = models.OneToOneField(User, on_delete=models.CASCADE, default=1)  # Ensure default value

    def __str__(self):
        return f"{self.user.username}"

class ContentCreator(models.Model):
    user = models.OneToOneField(User, on_delete=models.CASCADE, default=1)  # Ensure default value

    def __str__(self):
        return f"{self.user.username}"

class DataAnalyst(models.Model):
    user = models.OneToOneField(User, on_delete=models.CASCADE, default=1)  # Ensure default value

    def __str__(self):
        return f"{self.user.username}"

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
    profile_picture = models.ImageField(upload_to='profile_pictures/', null=True, blank=True)

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
    user = models.ForeignKey(UserAccount, on_delete=models.CASCADE)
    content = models.TextField()
    rating = models.PositiveIntegerField()  # 1 to 5 stars
    created_at = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return f"{self.user.user.username}'s Testimonial"


