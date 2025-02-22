# Generated by Django 3.1.12 on 2024-12-19 11:58

from django.conf import settings
from django.db import migrations, models
import django.db.models.deletion
import uuid

class Migration(migrations.Migration):

    initial = True

    dependencies = [
        migrations.swappable_dependency(settings.AUTH_USER_MODEL),
    ]

    operations = [
        migrations.CreateModel(
            name='Businessman',
            fields=[
                ('id', models.AutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('user', models.OneToOneField(on_delete=django.db.models.deletion.CASCADE, to=settings.AUTH_USER_MODEL, default=1)),  # Ensure default value
            ],
        ),
        migrations.CreateModel(
            name='Category',
            fields=[
                ('id', models.AutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('name', models.CharField(max_length=100)),
                ('description', models.TextField()),
                ('type', models.CharField(max_length=50, choices=[
                    ('content_creator', 'Content Creator'),
                    ('business', 'Business'),
                    ('analyst', 'Analyst'),
                    ('admin', 'Admin')  # Add 'admin' to choices
                ], default='admin')),  # Set default value to 'admin'
            ],
        ),
        migrations.CreateModel(
            name='ContentCreator',
            fields=[
                ('id', models.AutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('user', models.OneToOneField(on_delete=django.db.models.deletion.CASCADE, to=settings.AUTH_USER_MODEL, default=1)),  # Ensure default value
            ],
        ),
        migrations.CreateModel(
            name='Data',
            fields=[
                ('id', models.AutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('name', models.CharField(max_length=100)),
                ('age', models.IntegerField()),
                ('from_location', models.CharField(max_length=100)),
                ('klout_score', models.FloatField()),
            ],
        ),
        migrations.CreateModel(
            name='DataAnalyst',
            fields=[
                ('id', models.AutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('user', models.OneToOneField(on_delete=django.db.models.deletion.CASCADE, to=settings.AUTH_USER_MODEL, default=1)),  # Ensure default value
            ],
        ),
        migrations.CreateModel(
            name='UserAccount',
            fields=[
                ('id', models.AutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('user', models.OneToOneField(on_delete=django.db.models.deletion.CASCADE, to=settings.AUTH_USER_MODEL, default=1)),  # Ensure default value
                ('username', models.CharField(max_length=150, unique=True)),  # Add username field
                ('email', models.EmailField(max_length=254, unique=True)),  # Ensure email is unique
                ('role', models.CharField(max_length=255)),
            ],
        ),
        migrations.CreateModel(
            name='ExtractData',
            fields=[
                ('id', models.AutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('platform', models.CharField(max_length=100)),
                ('data', models.TextField()),
                ('extracted_at', models.DateTimeField(auto_now_add=True)),
                ('businessman', models.ForeignKey(on_delete=django.db.models.deletion.CASCADE, to=settings.AUTH_USER_MODEL)),
            ],
        ),
        migrations.CreateModel(
            name='Profile',
            fields=[
                ('id', models.AutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('profile_id', models.CharField(default=uuid.uuid4, max_length=100, unique=True)),
                ('first_name', models.CharField(max_length=100)),
                ('last_name', models.CharField(max_length=100)),
                ('company', models.CharField(max_length=100)),
                ('profile_picture', models.ImageField(blank=True, null=True, upload_to='profile_pictures/')),
            ],
        ),
        migrations.CreateModel(
            name='Testimonial',
            fields=[
                ('id', models.AutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('content', models.TextField()),
                ('rating', models.PositiveIntegerField()),
                ('created_at', models.DateTimeField(auto_now_add=True)),
                ('user', models.ForeignKey(on_delete=django.db.models.deletion.CASCADE, to='main.useraccount')),
            ],
        ),
        migrations.CreateModel(
            name='DataItem',
            fields=[
                ('id', models.AutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('name', models.CharField(max_length=255)),
                ('description', models.TextField(blank=True, null=True)),
                ('visibility', models.CharField(choices=[('private', 'Private'), ('restricted', 'Restricted'), ('public', 'Public')], default='private', max_length=20)),
                ('created_at', models.DateTimeField(auto_now_add=True)),
                ('businessman', models.ForeignKey(on_delete=django.db.models.deletion.CASCADE, to='main.useraccount')),
            ],
        ),
    ]
