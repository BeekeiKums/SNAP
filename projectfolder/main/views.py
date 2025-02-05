import base64
import csv
import io
import os
import re
from collections import defaultdict
from io import TextIOWrapper

import instaloader
import pandas as pd
from apify_client import ApifyClient
from django.conf import settings
from django.contrib import messages
from django.contrib.auth import authenticate, login as auth_login, logout as auth_logout
from django.contrib.auth.decorators import login_required
from django.http import HttpResponse, JsonResponse
from django.shortcuts import render, redirect, get_object_or_404
from django.views.decorators.csrf import csrf_exempt
from neomodel import db
import plotly.express as px
from plotly.io import to_html
from sklearn.calibration import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from neo4j import GraphDatabase
from django.contrib.auth.hashers import make_password
import requests
from django.contrib.auth.models import User

from .forms import (UserAccountForm, CategoryForm, BusinessmanForm, ContentCreatorForm, DataAnalystForm, ProfileForm, VisibilitySettingsForm, UserCreationForm)
from .models import Category, UserAccount, Profile, DataItem, Testimonial, Person, Movie

# Initialize Instaloader
L = instaloader.Instaloader()

def login_instagram(request):
    if request.method == "POST":
        username = request.POST.get("username")
        password = request.POST.get("password")
        try:
            L.login(username, password)
            return JsonResponse({"success": True})
        except Exception as e:
            return JsonResponse({"success": False, "error": str(e)})

def load_session():
    """Load an existing session or log in manually."""
    session_file = os.path.join(settings.BASE_DIR, "instaloader_session")
    if os.path.exists(session_file):
        try:
            L.load_session_from_file(username=None, filename=session_file)
        except Exception:
            login(session_file)
    else:
        login(session_file)

def login(session_file):
    """Log in to Instagram and save the session."""
    username = os.getenv("INSTAGRAM_USERNAME", "default_username")
    password = os.getenv("INSTAGRAM_PASSWORD", "default_password")
    L.login(username, password)
    L.save_session_to_file(session_file)

def extract_hashtags(caption):
    """Extract hashtags from a caption."""
    if not caption:
        return []
    return re.findall(r"#(\w+)", caption)

def scrape_profile(request):
    if request.method == "POST":
        profile_username = request.POST.get("profile_username")
        num_posts = int(request.POST.get("num_posts", 10))  # Default to 10 posts if no input is given

        if not profile_username:
            messages.error(request, "Profile username is required.")
            return redirect('scrape_profile')

        try:
            # Authenticate if necessary
            if not L.context.is_logged_in:
                L.login("your_instagram_username", "your_password")

            # Load profile using Instaloader
            profile = instaloader.Profile.from_username(L.context, profile_username)

            # Define a dynamic path for saving the CSV file
            csv_folder = os.path.join(settings.MEDIA_ROOT, "downloads")
            os.makedirs(csv_folder, exist_ok=True)  # Create the downloads folder if it doesn't exist
            csv_filename = os.path.join(csv_folder, f"{profile_username}_posts.csv")

            # Collect post data
            posts_data = []
            for i, post in enumerate(profile.get_posts()):
                if i >= num_posts: 
                    break

                try:
                    # Extract specific attributes, location cannot be extracted unfortunately
                    post_details = {
                        "owner_username": post.owner_username,
                        "is_verified": "YES" if profile.is_verified else "NO",
                        "followers": profile.followers,
                        "shortcode": post.shortcode,
                        "timestamp": post.date_utc.strftime("%Y-%m-%d %H:%M:%S"),
                        "title": post.title,                       
                        "caption": post.caption,
                        "likes": post.likes,
                        "comments": post.comments,
                        "hashtags": post.caption_hashtags,  # List of hashtags
                        "is_video": post.is_video,
                        "video_url": post.video_url if post.is_video else None,
                        "video_duration": post.video_duration,
                        "image_url": post.url if not post.is_video else None,
                        "is_sponsored": post.is_sponsored,
                    }

                    posts_data.append(post_details)
                except Exception as e:
                    messages.error(request, f"Error processing post: {str(e)}")
                    return redirect('scrape_profile')

            # Prepare fieldnames for the selected attributes
            fieldnames = [
                "owner_username",
                "is_verified",
                "followers",
                "shortcode",
                "timestamp",
                "title",
                "caption",
                "likes",
                "comments",
                "hashtags",
                "is_video",
                "video_url",
                "video_duration",
                "image_url",
                "is_sponsored"
            ]

            # Write to CSV
            with open(csv_filename, mode="w", newline="", encoding="utf-8") as csvfile:
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(posts_data)

            # Return success response with file URL
            file_url = os.path.join(settings.MEDIA_URL, "downloads", f"{profile_username}_posts.csv")
            messages.success(request, f"Profile scraped successfully. Download CSV: {file_url}")
            return redirect('scrape_profile')

        except Exception as e:
            messages.error(request, f"Error scraping profile: {str(e)}")
            return redirect('scrape_profile')

    return render(request, "main/scraper.html")

# ...existing code...

def login(request):
    if request.method == 'POST':
        username = request.POST.get('username')
        password = request.POST.get('password')
        
        # Use Django's built-in authentication system
        try:
            user = User.objects.get(username=username)
            try:
                user_account = UserAccount.objects.filter(user=user).first()
                if not user_account:
                    messages.error(request, "User account does not exist!")
                    return redirect('login')
            except UserAccount.DoesNotExist:
                messages.error(request, "User account does not exist!")
                return redirect('login')
            
            if user.check_password(password):  # Use Django's built-in password check
                # Set session for login
                request.session['username'] = user.username
                request.session['role'] = user_account.role
                request.session['is_authenticated'] = True

                # Debug statements
                print(f"User {username} authenticated successfully.")
                print(f"Role: {user_account.role}")

                # Redirect to respective dashboards based on role
                if user_account.role == 'admin':
                    return redirect('dashboard')
                elif user_account.role == 'businessman':
                    return redirect('businessman_dashboard')
                elif user_account.role == 'content_creator':
                    return redirect('content_creator_dashboard')
                elif user_account.role == 'data_analyst':
                    return redirect('data_analyst_dashboard')
                else:
                    messages.error(request, "Invalid role!")
                    print("Invalid role!")
                    return redirect('login')
            else:
                messages.error(request, "Invalid credentials!")
                print("Invalid credentials!")
        except User.DoesNotExist:
            messages.error(request, "Account does not exist!")
            print("Account does not exist!")

    return render(request, 'main/login.html')

def logout(request):
    auth_logout(request)
    return redirect('login')

def businessman_dashboard(request):
    return render(request, 'main/businessman.html')

def content_creator_dashboard(request):
    return render(request, 'main/content_creator.html')

def data_analyst_dashboard(request):
    return render(request, 'main/data_analyst.html')

# Dashboard View
def dashboard(request):
    return render(request, 'main/dashboard.html')

# Create Category
def create_category(request):
    if request.method == 'POST':
        form = CategoryForm(request.POST)
        if form.is_valid():
            form.save() 
            return redirect('category_list')
    else: 
        form = CategoryForm()  # Empty form for GET request
    return render(request, 'main/create_category.html', {'form': form})

@csrf_exempt
def update_category(request, category_id):
    if request.method == 'POST':
        name = request.POST.get('name')
        description = request.POST.get('description')
        type = request.POST.get('type')

        try:
            category = Category.objects.get(id=category_id)
            category.name = name
            category.description = description
            category.type = type
            category.save()

            # Return both the updated name and description
            return JsonResponse({
                'status': 'success',
                'id': category.id,
                'name': category.name,
                'description': category.description,
                'type': category.type
            })
        except Category.DoesNotExist:
            return JsonResponse({'status': 'error', 'message': 'Category not found'})

    return JsonResponse({'status': 'error', 'message': 'Invalid request'})

def category_list(request):
    categories = Category.objects.all()  # Make sure this is fetching the categories correctly
    return render(request, 'main/category_list.html', {'categories': categories})

def get_timezone_from_ip(ip):
    try: 
        # Using ip-api to get timezone
        response = requests.get(f"http://ip-api.com/json/{ip}")
        if response.status_code == 200:
            data = response.json()
            return data.get("timezone", "Unknown")
    except Exception as e:
        print(f"Error fetching timezone: {e}")
    return "Unknown"

def create_profile(request):
    if request.method == 'POST':
        form = ProfileForm(request.POST, request.FILES)
        if form.is_valid():
            # Save the form data to create a new profile instance
            form.save()
            # Redirect to the login page after profile creation
            messages.success(request, 'Profile created successfully! Please log in.')
            return redirect('login')
    else:
        form = ProfileForm()
        
    # Render the template with the form
    return render(request, 'main/create_profile.html', {'form': form})

def get_client_ip(request):
    """ Extract the client's IP from the request headers"""
    x_forwarded_for = request.META.get('HTTP_X_FORWARDED_FOR')
    if x_forwarded_for:
        ip = x_forwarded_for.split(',')[0]
    else:
        ip = request.META.get('REMOTE_ADDR')
    return ip

# ...existing code...

def create_user_account(request):
    if request.method == 'POST':
        form = UserCreationForm(request.POST)
        if form.is_valid():
            username = form.cleaned_data.get('username')
            password = form.cleaned_data.get('password1')
            email = form.cleaned_data.get('email')
            role = request.POST.get('role')
            try:
                user = User.objects.create_user(username=username, password=password, email=email)
                UserAccount.objects.create(user=user, username=username, role=role, email=email)
                messages.success(request, 'Account created successfully.')
                return redirect('login')
            except Exception as e:
                messages.error(request, f'Error creating account: {e}')
        else:
            messages.error(request, 'Invalid form submission.')
    else:
        form = UserCreationForm()
    return render(request, 'main/create_user_account.html', {'form': form})

def create_businessman_account(request):
    if request.method == 'POST':
        form = BusinessmanForm(request.POST)
        if form.is_valid():
            if UserAccount.objects.filter(username=form.cleaned_data['username']).exists():
                messages.error(request, "Username already exists!")
            elif UserAccount.objects.filter(email=form.cleaned_data['email']).exists():
                messages.error(request, "Email already exists!")
            else:
                user_account = form.save(commit=False)
                user_account.role = 'businessman'
                user_account.password = make_password(form.cleaned_data['password'])
                user_account.save()
                messages.success(request, "Businessman account created successfully!")
                return redirect('create_businessman_account')
    else:
        form = BusinessmanForm()
    return render(request, 'main/create_businessman_acc.html', {'form': form})

def create_content_creator_account(request):
    if request.method == 'POST':
        form = ContentCreatorForm(request.POST)
        if form.is_valid():
            if UserAccount.objects.filter(username=form.cleaned_data['username']).exists():
                messages.error(request, "Username already exists!")
            elif UserAccount.objects.filter(email=form.cleaned_data['email']).exists():
                messages.error(request, "Email already exists!")
            else:
                user_account = form.save(commit=False)
                user_account.role = 'content_creator'
                user_account.password = make_password(form.cleaned_data['password'])
                user_account.save()
                messages.success(request, "Content creator account created successfully!")
                return redirect('create_content_creator_account')
    else:
        form = ContentCreatorForm()
    return render(request, 'main/create_content_creator_acc.html', {'form': form})

def create_data_analyst_account(request):
    if request.method == 'POST':
        form = DataAnalystForm(request.POST)
        if form.is_valid():
            if UserAccount.objects.filter(username=form.cleaned_data['username']).exists():
                messages.error(request, "Username already exists!")
            elif UserAccount.objects.filter(email=form.cleaned_data['email']).exists():
                messages.error(request, "Email already exists!")
            else:
                user_account = form.save(commit=False)
                user_account.role = 'data_analyst'
                user_account.password = make_password(form.cleaned_data['password'])
                user_account.save()
                messages.success(request, "Data analyst account created successfully!")
                return redirect('create_data_analyst_account')
    else:
        form = DataAnalystForm()
    return render(request, 'main/create_data_analyst_acc.html', {'form': form})

# ...existing code...

# View user profile
def view_profile(request):
    user_profile = Profile.objects.all()
    return render(request, 'main/myprofile.html', {'user_profile': user_profile})

# View to list all user accounts
def view_user_accounts(request):
    users = UserAccount.objects.all()
    return render(request, 'main/view_user_accounts.html', {'users': users})

def view_businessman_accounts(request):
    users = UserAccount.objects.filter(role='businessman')
    return render(request, 'main/view_businessman_accounts.html', {'users': users})

def view_content_creator_accounts(request):
    users = UserAccount.objects.filter(role='content_creator')
    return render(request, 'main/view_content_accounts.html', {'users': users})

def view_data_analyst_accounts(request):
    users = UserAccount.objects.filter(role='data_analyst')
    return render(request, 'main/view_data_analyst_accounts.html', {'users': users})

def update_user_account(request, user_id):
    if request.method == 'POST':
        user = get_object_or_404(UserAccount, id=user_id)
        user.username = request.POST.get('username')
        user.email = request.POST.get('email')
        user.save()
        
        return JsonResponse({'status': 'success'})

    return JsonResponse({'status': 'fail'})

def view_profile_content_creator(request):
    return render(request, 'view_profile_content_creator.html')

import json

def update_profile(request, profile_id):
    if request.method == 'POST':
        profile = get_object_or_404(Profile, profile_id=profile_id)
        data = json.loads(request.body)  # Parse JSON data
        profile.first_name = data.get('first_name', profile.first_name)
        profile.last_name = data.get('last_name', profile.last_name)
        profile.company = data.get('company', profile.company)
        profile.timezone = data.get('timezone', profile.timezone)
        
        profile.save()
        
        return JsonResponse({'status': 'success'})

    return JsonResponse({'status': 'fail'}, status=400)

def upload_csv_or_xlsx(request):
    if request.method == "POST":
        uploaded_file = request.FILES.get("file")

        if not uploaded_file:
            messages.error(request, "Error: No file uploaded.")
            return redirect('upload_csv_or_xlsx')

        try:
            # Save the original file name in the session
            original_file_name = os.path.splitext(uploaded_file.name)[0]
            cleaned_file_name = original_file_name.strip() + ".csv"
            request.session['original_file_name'] = cleaned_file_name

            # Read CSV file
            file_data = TextIOWrapper(uploaded_file.file, encoding='utf-8')
            df = pd.read_csv(file_data)

            # Clean and convert numeric columns
            for col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')  # Convert to numeric where possible
            
            # Replace NaNs with a placeholder or drop them
            df = df.dropna(axis=0, how='all')  # Drop rows where all values are NaN

            # Extract headers and data for later use
            headers = df.columns.tolist()
            data = df.to_dict(orient='records')  # Convert cleaned DataFrame to a list of dictionaries

            if not headers:
                messages.error(request, "The CSV file must have at least one column.")
                return redirect('upload_csv_or_xlsx')

            # Save headers and data to session
            request.session['uploaded_headers'] = headers
            request.session['uploaded_data'] = data
            messages.success(request, f"File {cleaned_file_name} uploaded successfully.")
            return redirect('auto_preprocess')
        except Exception as e:
            messages.error(request, f"Error processing the file: {str(e)}")
            return redirect('upload_csv_or_xlsx')

    return render(request, "main/upload_excel.html")

def auto_preprocess(request):
    import pandas as pd
    from sklearn.preprocessing import LabelEncoder

    if request.method == 'POST':
        try:
            # Retrieve uploaded data from the session
            uploaded_data = request.session.get('uploaded_data', [])
            if not uploaded_data:
                messages.error(request, "No data available for preprocessing. Please upload a file first.")
                return redirect('upload_csv_or_xlsx')

            # Convert session data to DataFrame
            df = pd.DataFrame(uploaded_data)

            # Step 1: Drop rows and columns where all values are NaN
            df.dropna(how='all', inplace=True)
            df.dropna(axis=1, how='all', inplace=True)

            # Step 2: Clean column names dynamically
            df.columns = (
                df.columns.str.strip()
                .str.replace(r'\W+', '_', regex=True)
                .str.lower()
            )
            
            # Remove columns with a single unique value (constant columns)
            df = df.loc[:, df.nunique() > 1]

            # Remove columns with > 90% missing values
            df = df.loc[:, df.isnull().mean() < 0.9]

            # Step 3: Process columns dynamically
            processed_columns = []
            ignored_columns = []
            label_encoder = LabelEncoder()  # For encoding categories
            max_unique_values = 50  # Limit unique values to avoid graph clutter

            for col in df.columns:
                if df[col].dtype == 'object':  # Handle text columns
                    # Remove unnecessary whitespace and invalid strings
                    df[col] = df[col].astype(str).str.strip().replace({'nan': '', 'NaN': '', 'None': ''})

                    # If possible, convert to numeric
                    df[col] = pd.to_numeric(df[col], errors='coerce')

                    # If still non-numeric, encode the column
                    if df[col].isnull().all():  # Still non-numeric
                        df[col] = label_encoder.fit_transform(df[col].astype(str))
                        processed_columns.append(col)

                elif pd.api.types.is_numeric_dtype(df[col]):  # Handle numeric columns
                    df[col].fillna(0, inplace=True)
                    processed_columns.append(col)

                else:
                    ignored_columns.append(col)

            # Step 4: Sample rows for performance
            if len(df) > 500:  # Adjust sampling threshold as needed
                df = df.sample(n=500, random_state=42)

            # Save cleaned data back to the session
            request.session['uploaded_data'] = df.to_dict(orient='records')
            request.session['uploaded_headers'] = df.columns.tolist()

            messages.success(request, "Data successfully preprocessed. All columns converted to numeric where possible.")
            return redirect('manual_preprocess')

        except Exception as e:
            messages.error(request, f"Error during preprocessing: {str(e)}")
            return redirect('upload_csv_or_xlsx')

    return render(request, 'main/data_management.html')

def manual_preprocess(request):
    import math

    # Pagination settings
    PAGE_SIZE = 100  # Number of rows per page
    if request.method == 'POST':
        try:
            # Retrieve uploaded data from the session
            uploaded_data = request.session.get('uploaded_data', [])
            headers = request.session.get('uploaded_headers', [])

            # Update rows with submitted data
            for index, row in enumerate(uploaded_data):
                for header in headers:
                    field_name = f'data_{index}_{header}'
                    row[header] = request.POST.get(field_name, row.get(header, ''))

            # Save updated data back to the session
            request.session['uploaded_data'] = uploaded_data
            messages.success(request, "Data successfully updated.")
            return redirect('create_or_view_visualization')
        except Exception as e:
            messages.error(request, f"An error occurred: {str(e)}")
            return redirect('manual_preprocess')

    # Retrieve data for display
    uploaded_data = request.session.get('uploaded_data', [])
    headers = request.session.get('uploaded_headers', [])
    
    # Implement pagination
    page = int(request.GET.get('page', 1))
    total_pages = math.ceil(len(uploaded_data) / PAGE_SIZE)
    paginated_data = uploaded_data[(page -1) * PAGE_SIZE : page * PAGE_SIZE]

    return render(request, 'main/manual.html', {'data': paginated_data, 
                                                'headers': headers,
                                                'current_page': page,
                                                'total_pages': total_pages})

def dashboard_view(request):
    # Retrieve headers and data from session
    headers = request.session.get('uploaded_headers', [])
    uploaded_data = request.session.get('uploaded_data', [])

    if not headers or not uploaded_data:
        messages.error(request, "No data uploaded. Please upload a file first.")
        return redirect('upload_csv_or_xlsx')

    # Filter numeric columns only
    df = pd.DataFrame(uploaded_data)
    numeric_headers = [col for col in headers if pd.api.types.is_numeric_dtype(pd.to_numeric(df[col], errors='coerce'))]

    return render(request, 'main/dash_visualization.html', {
        'headers': numeric_headers
    })

from django.shortcuts import render
import networkx as nx
import plotly.graph_objects as go

def create_or_view_visualization(request):
    try:
        # Retrieve uploaded data
        uploaded_data = request.session.get('uploaded_data', [])
        headers = request.session.get('uploaded_headers', [])

        if not uploaded_data or not headers:
            messages.error(request, "No data available for visualization.")
            return redirect('upload_csv_or_xlsx')

        # Step 1: Build the Graph
        G = nx.Graph()
        for row in uploaded_data:
            row_nodes = []
            for header in headers:
                value = row.get(header)
                if value and isinstance(value, str) and len(value) < 50 and value not in ['null', 'None', 'nan', '-', '']:
                    if 'http' in value or 'timestamp' in header.lower():
                        continue
                    G.add_node(value)
                    row_nodes.append(value)

            # Add edges with dynamic limit
            max_edges_per_node = 10
            node_edges_count = {}
            for i in range(len(row_nodes)):
                for j in range(i + 1, len(row_nodes)):
                    node1, node2 = row_nodes[i], row_nodes[j]
                    if node_edges_count.get(node1, 0) >= max_edges_per_node:
                        continue
                    if node_edges_count.get(node2, 0) >= max_edges_per_node:
                        continue
                    G.add_edge(node1, node2)
                    node_edges_count[node1] = node_edges_count.get(node1, 0) + 1
                    node_edges_count[node2] = node_edges_count.get(node2, 0) + 1

        # Step 2: Calculate Centrality Metrics
        degree_centrality = nx.degree_centrality(G)
        closeness_centrality = nx.closeness_centrality(G)
        betweenness_centrality = nx.betweenness_centrality(G)

        # Combine centrality measures into a dictionary
        centrality_metrics = {}
        for node in G.nodes():
            centrality_metrics[node] = {
                "degree": degree_centrality.get(node, 0),
                "closeness": closeness_centrality.get(node, 0),
                "betweenness": betweenness_centrality.get(node, 0)
            }

        # Identify top nodes by degree centrality
        top_degree_nodes = sorted(degree_centrality, key=degree_centrality.get, reverse=True)[:10]

        # Step 3: Dynamic Layout
        layout_style = request.GET.get('layout', 'kamada_kawai')
        if layout_style == 'spring':
            pos = nx.spring_layout(G, k=0.5, seed=42)  # Adjust "k" for spacing
        elif layout_style == 'circular':
            pos = nx.circular_layout(G)
        else:
            pos = nx.kamada_kawai_layout(G)  # Kamada-Kawai for better large graph visualization

        edge_x = []
        edge_y = []
        for edge in G.edges():
            x0, y0 = pos[edge[0]]
            x1, y1 = pos[edge[1]]
            edge_x.extend([x0, x1, None])
            edge_y.extend([y0, y1, None])

        node_x = []
        node_y = []
        node_text = []
        for node in G.nodes():
            x, y = pos[node]
            node_x.append(x)
            node_y.append(y)
            node_text.append(f"Node: {node}<br>Degree: {degree_centrality[node]:.2f}")

        # Edge trace for plotly
        edge_trace = go.Scatter(
            x=edge_x, y=edge_y,
            line=dict(width=0.5, color='#888'),
            hoverinfo='none',
            mode='lines'
        )

        # Node trace for plotly
        node_trace = go.Scatter(
            x=node_x, y=node_y,
            mode='markers+text',
            text=node_text,
            textposition='top center',
            marker=dict(
                size=[degree_centrality[node] * 20 + 10 for node in G.nodes()],
                color=['orange' if node in top_degree_nodes else 'skyblue' for node in G.nodes()],
                line_width=2
            ),
            hoverinfo='text'
        )

        fig = go.Figure(data=[edge_trace, node_trace],
                        layout=go.Layout(
                            title='Interactive Network Visualization',
                            showlegend=False,
                            hovermode='closest',
                            margin=dict(b=0, l=0, r=0, t=40),
                            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False)
                        ))
        
        # Prepare centrality results for frontend display
        centrality_results = [
            {"node": node, "degree": round(metrics["degree"], 4), 
             "closeness": round(metrics["closeness"], 4), 
             "betweenness": round(metrics["betweenness"], 4)}
            for node, metrics in sorted(centrality_metrics.items(), key=lambda x: x[1]["degree"], reverse=True)[:10]
        ]

        # Convert to HTML
        graph_html = fig.to_html(full_html=False)
        
        # Determine the template
        template = 'main/businessman.html' if request.GET.get('view') == 'businessman' else 'main/view_visualization.html'

        # Render visualization
        return render(request, template, {
            'graph_html': graph_html,
            'node_count': G.number_of_nodes(),
            'edge_count': G.number_of_edges(),
            'headers': headers,
            'centrality_results': centrality_results  # Top nodes with centrality scores
        })

    except Exception as e:
        messages.error(request, f"Error creating visualization: {str(e)}")
        return redirect('upload_csv_or_xlsx')

def test_predictive_models(request):
    try:
        # Retrieve uploaded data
        uploaded_data = request.session.get('uploaded_data', [])
        headers = request.session.get('uploaded_headers', [])

        if not uploaded_data or not headers:
            messages.error(request, "No data available for model testing. Please upload data.")
            return redirect('upload_csv_or_xlsx')

        # Load data into DataFrame
        df = pd.DataFrame(uploaded_data)
        print("Original DataFrame:")
        print(df.dtypes)
        print(df.head())
        
        # User inputs from form
        target_column = request.GET.get('target_column')
        model_type = request.GET.get('model_type', 'linear_regression')

        # Input validation: Target column exists
        if not target_column or target_column not in df.columns:
            messages.error(request, "Please select a valid target column.")
            return redirect('create_or_view_visualization')
        
        # Step 1: Convert target column to numeric
        df[target_column] = pd.to_numeric(df[target_column], errors='coerce')
        if df[target_column].isnull().all():
            messages.error(request, f"Target column '{target_column}' is not numeric and cannot be converted.")
            return redirect('create_or_view_visualization')

        # Step 2: Handle non-numeric features (encode them)
        numeric_df = df.copy()
        label_encoder = LabelEncoder()

        for col in numeric_df.columns:
            if col != target_column and numeric_df[col].dtype == 'object':
                print(f"Encoding column: {col}")
                numeric_df[col] = numeric_df[col].fillna('Missing')  # Replace NaNs with a placeholder
                numeric_df[col] = label_encoder.fit_transform(numeric_df[col].astype(str))
            numeric_df[col] = pd.to_numeric(numeric_df[col], errors='coerce')  # Ensure numeric conversion

        # Drop rows with missing target values
        numeric_df = numeric_df.dropna(subset=[target_column])

        # Define features and target
        X = numeric_df.drop(columns=[target_column])
        y = numeric_df[target_column]
        
        # Check if there are valid features left
        if X.empty or y.empty:
            messages.error(request, "No valid features left for modeling after preprocessing.")
            return redirect('create_or_view_visualization')

        # Train-test split
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Run selected model
        metrics = {}
        if model_type == 'linear_regression':
            model = LinearRegression()
        elif model_type == 'decision_tree':
            model = DecisionTreeRegressor()
        elif model_type == 'random_forest':
            model = RandomForestRegressor()   
        else:
            messages.error(request, "Invalid model type selected.")
            return redirect('create_or_view_visualization')

        # Train and predict
        model.fit(X_train, y_train)
        predictions = model.predict(X_test)

        # Calculate metrics
        metrics = {
            "Mean Squared Error": mean_squared_error(y_test, predictions),
            "R2 Score": r2_score(y_test, predictions)
        }

        # Plot the predictions
        buf = io.BytesIO()
        plt.figure(figsize=(10, 6))
        plt.plot(y_test.values[:50], label="Actual", marker='o')
        plt.plot(predictions[:50], label="Predicted", marker='x')
        plt.legend()
        plt.title(f"{model_type.replace('_', ' ').title()} - Predicted vs Actual")
        plt.savefig(buf, format='png', bbox_inches='tight')
        plt.close()
        buf.seek(0)
        img_base64 = base64.b64encode(buf.read()).decode('utf-8')

        # Render the predictive model results
        return render(request, 'main/predictive.html', {
            'metrics': metrics,
            'prediction_plot': f"data:image/png;base64,{img_base64}",
            'model_type': model_type
        })

    except Exception as e:
        messages.error(request, f"Model testing failed: {str(e)}")
        return redirect('create_or_view_visualization')

def save_to_neo4j(graph, headers):

    import logging
    #Neo4j connection details
    uri = "bolt://localhost:7687"
    username = "neo4j"
    password = "bin754826" 
    
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    
    driver = GraphDatabase.driver(uri, auth=(username, password))
    
    def add_to_neo4j(tx, node1,node2):
        query = """
        MERGE (n1:Node {name: $node1})
        MERGE (n2:Node {name: $node2})
        MERGE (n1)-[r:RELATION]->(n2)
        """
        
        tx.run(query, node1=node1, node2=node2)
        
    with driver.session() as session:
        for edge in graph.edges:
            session.write_transaction(add_to_neo4j, edge[0], edge[1])
            
    driver.close()   
    logger.info("All data saved to Neo4j successfully.")


from django.shortcuts import render
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from io import BytesIO
import base64
import calendar

# Use the 'Agg' backend for Matplotlib
import matplotlib
matplotlib.use('Agg')

def upload_and_view_charts(request):
    fig_html_list = []
    if request.method == 'POST' and 'csv_file' in request.FILES:
        try:
            csv_file = request.FILES['csv_file']
            platform = request.POST.get('platform')
            sponsored = request.POST.get('sponsored')
            post_type = request.POST.get('post_type')
            time_duration = request.POST.get('time_duration')
            df = pd.read_csv(csv_file)

            platform_handlers = {
                'instagram': handle_instagram_data
            }

            handler = platform_handlers.get(platform)
            if handler:
                fig_html_list = handler(df, sponsored, post_type, time_duration)
            else:
                fig_html_list = [f"<div>Error: Unknown platform '{platform}'</div>"]
        except Exception as e:
            fig_html_list = [f"<div>Error processing the file: {str(e)}</div>"]

    return render(request, 'main/upload_and_view_charts.html', {'fig_html_list': fig_html_list})

def handle_instagram_data(csv_raw, sponsored, post_type, time_duration):
    fig_html_list = []

    # Clean the 'Likes' column by converting to numeric
    csv_raw['Likes'] = pd.to_numeric(csv_raw['Likes'], errors='coerce')

    # Handle missing values in 'Hour'
    csv_raw = csv_raw.dropna(subset=['Hour'])

    # Convert numeric month to month name
    csv_raw['Month'] = csv_raw['Month'].apply(lambda x: calendar.month_name[int(x)])

    # Filter data based on selections
    if sponsored != 'all':
        csv_raw = csv_raw[csv_raw['Sponsored'] == (sponsored == 'yes')]
    if post_type != 'all':
        csv_raw = csv_raw[csv_raw['Is Video'] == (post_type == 'video')]

    # Define time categories and ordering
    order_days = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    order_months = list(calendar.month_name)[1:]  # Skip the empty string at index 0
    time_categories = {
        "month": ("Month", 'Month', order_months),
        "day_of_week": ("Day of the Week", 'Day of Upload', order_days),
        "hour": ("Hour of the Day", 'Hour', None)
    }

    time_name, time_col, time_order = time_categories[time_duration]

    # Define color palettes
    color_palettes = [
        'Set1', 'Set2', 'viridis', 'coolwarm', 'pastel',
        'deep', 'muted', 'dark', 'colorblind', 'cubehelix'
    ]
    
    # Generate plots for the selected category and time category
    palette_idx = 0
    palette = sns.color_palette(color_palettes[palette_idx % len(color_palettes)])
    palette_idx += 1

    # Ensure the time column is correctly ordered if necessary
    if time_order:
        csv_raw.loc[:, time_col] = pd.Categorical(csv_raw[time_col], categories=time_order, ordered=True)

    # Create count plot for the selected category and time feature
    count_title = f'Number of Posts by {time_name}'
    count_xlabel = time_name
    count_ylabel = 'Number of Posts'
    count_plot_html = create_countplot(csv_raw, time_col, count_title, count_xlabel, count_ylabel, order=time_order, palette=palette)
    fig_html_list.append(count_plot_html)

    # Create engagement plot (likes vs comments) for the selected category and time feature
    grouped_data = csv_raw.groupby(time_col, observed=True)[['Likes', 'Comments']].mean().reset_index()
    engagement_title = f'Average Engagement by {time_name}'
    engagement_xlabel = time_name
    engagement_ylabel = 'Average Engagement'
    engagement_plot_html = create_engagement_plot(grouped_data, time_col, engagement_title, engagement_xlabel, engagement_ylabel)
    fig_html_list.append(engagement_plot_html)

    return fig_html_list

def create_engagement_plot(data, x, title, xlabel, ylabel, ax=None):
    """
    Creates a line plot comparing likes and comments over time for engagement analysis.
    """
    fig, ax = plt.subplots(figsize=(12, 6)) if ax is None else (fig, ax)
    sns.lineplot(x=x, y='Likes', data=data, label='Likes', marker='o', ax=ax)
    sns.lineplot(x=x, y='Comments', data=data, label='Comments', marker='o', ax=ax)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.legend()
    ax.grid()
    plot_html = plot_to_html(fig)
    plt.close(fig)
    return plot_html

def create_countplot(data, time_col, title, xlabel, ylabel, order=None, palette="Set1"):
    """
    Creates a count plot for a given time column and returns the plot as HTML.
    """
    fig, ax = plt.subplots(figsize=(10, 5))
    sns.countplot(x=time_col, data=data, ax=ax, order=order, palette=palette)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    plt.xticks(rotation=45)
    plot_html = plot_to_html(fig)
    plt.close(fig)
    return plot_html

def plot_to_html(fig):
    """
    Convert a Matplotlib figure to an HTML img tag with responsive styling.
    """
    buffer = BytesIO()
    fig.savefig(buffer, format='png', bbox_inches='tight', dpi=150)  # Adjust dpi for better resolution
    buffer.seek(0)
    img_str = base64.b64encode(buffer.read()).decode('utf-8')
    buffer.close()
    return f"<img src='data:image/png;base64,{img_str}' style='max-width: 100%; height: auto;'/>"
'''
def upload_and_view_charts(request):
    fig_html_list = []
    if request.method == 'POST' and 'csv_file' in request.FILES:
        csv_file = request.FILES['csv_file']
        platform = request.POST.get('platform', 'instagram')
        try:

            df = pd.read_csv(csv_file)

            platform_handlers = {
                'tiktok': handle_tiktok_data,
                'linkedin': handle_linkedin_data,
                'instagram': handle_instagram_data
            }

            handler = platform_handlers.get(platform)
            if handler:
                fig_html_list = handler(df)
            else:
                fig_html_list = [f"<div>Error: Unknown platform '{platform}'</div>"]
        except Exception as e:
            fig_html_list = [f"<div>Error processing the file: {str(e)}</div>"]

    return render(request, 'main/upload_and_view_charts.html', {'fig_html_list': fig_html_list})

def handle_instagram_data(df):
    fig_html_list = []

    # Ensure followers_count is numeric for Chart 1
    if 'followers_count' in df.columns:
        # Convert to numeric and drop invalid entries
        df['followers_count'] = pd.to_numeric(df['followers_count'], errors='coerce')
        valid_followers_df = df.dropna(subset=['followers_count'])
        fig1 = px.histogram(
            valid_followers_df,
            x='followers_count',
            title='Distribution of Followers',
            labels={'followers_count': 'Followers Count'},
            nbins=20  # Adjust number of bins as needed
        )
        fig_html_list.append(fig1.to_html(full_html=False))

    # Ensure likes_count and comments_count are numeric for Chart 2
    if {'account_type', 'likes_count', 'comments_count'}.issubset(df.columns):
        # Convert to numeric and drop invalid entries
        df['likes_count'] = pd.to_numeric(df['likes_count'], errors='coerce')
        df['comments_count'] = pd.to_numeric(df['comments_count'], errors='coerce')

        # Remove rows with NaN values in required columns
        valid_interaction_df = df.dropna(subset=['account_type', 'likes_count', 'comments_count'])

        # Group by account_type and calculate the mean
        interaction_avg = (
            valid_interaction_df
            .groupby('account_type')
            .mean(numeric_only=True)[['likes_count', 'comments_count']]  # Ensure only numeric columns are aggregated
            .reset_index()
        )

        fig2 = px.bar(
            interaction_avg,
            x='account_type',
            y=['likes_count', 'comments_count'],
            title='Average Interaction by Account Type',
            labels={'value': 'Average Count', 'account_type': 'Account Type', 'variable': 'Metric'},
            barmode='group'  # Grouped bar chart
        )
        fig_html_list.append(fig2.to_html(full_html=False))

    # Chart 3: Histogram
    if 'engagement_rate' in df.columns:
        fig3 = px.histogram(
            df,
            x='engagement_rate',
            title='Post interaction rate distribution',
            labels={'engagement_rate': 'engagement_rate'},
            nbins=20
        )
        fig_html_list.append(fig3.to_html(full_html=False))

    # Chart 4: User activity active time period (Heatmap)
    if 'post_timestamp' in df.columns and 'likes_count' in df.columns:
        df['hour'] = df['post_timestamp'].dt.hour
        heatmap_data = df.groupby(['hour'])['likes_count'].sum().reset_index()
        fig4 = px.density_heatmap(
            heatmap_data,
            x='hour',
            y='likes_count',
            title='User activity active time period',
            labels={'hour': 'hour', 'likes_count': 'likes_count'}
        )
        fig_html_list.append(fig4.to_html(full_html=False))

    # Chart 5: relationship likes_count and comments on a post (Scatter Plot)
    if {'likes_count', 'comments_count'}.issubset(df.columns):
        fig5 = px.scatter(
            df,
            x='likes_count',
            y='comments_count',
            title='relationship likes_count and comments on a post',
            labels={'likes_count': 'likes_count', 'comments_count': 'comments_count'}
        )
        fig_html_list.append(fig5.to_html(full_html=False))

    return fig_html_list

def handle_tiktok_data(df):
    fig_html_list = []

    # Chart 1: Likes count over time (Line Chart)
    if 'post_timestamp' in df.columns and 'likes_count' in df.columns:
        # Ensure the timestamp and likes count columns are valid
        df['post_timestamp'] = pd.to_datetime(df['post_timestamp'], errors='coerce')
        df['likes_count'] = pd.to_numeric(df['likes_count'], errors='coerce')

        # Remove NaN values from the timestamp and likes count columns
        valid_likes_df = df.dropna(subset=['post_timestamp', 'likes_count'])

        # Sort by timestamp and generate the line chart
        fig_likes_trend = px.line(
            valid_likes_df.sort_values('post_timestamp'),
            x='post_timestamp',
            y='likes_count',
            title='Likes Count Over Time',
            labels={'post_timestamp': 'Time', 'likes_count': 'Likes Count'}
        )
        fig_html_list.append(to_html(fig_likes_trend, full_html=False))

    # Chart 2: Engagement rate distribution (Histogram)
    if 'engagement_rate' in df.columns:
        df['engagement_rate'] = pd.to_numeric(df['engagement_rate'], errors='coerce')
        valid_engagement_df = df.dropna(subset=['engagement_rate'])
        fig3 = px.histogram(
            valid_engagement_df,
            x='engagement_rate',
            title='Engagement Rate Distribution',
            labels={'engagement_rate': 'Engagement Rate'},
            nbins=20  # Adjust the number of bins as needed
        )
        fig_html_list.append(fig3.to_html(full_html=False))

    # Chart 3: User activity by time period (Heatmap)
    if 'post_timestamp' in df.columns and 'likes_count' in df.columns:
        df['hour'] = df['post_timestamp'].dt.hour
        heatmap_data = df.groupby(['hour'])['likes_count'].sum().reset_index()
        fig4 = px.density_heatmap(
            heatmap_data,
            x='hour',
            y='likes_count',
            title='User Activity by Time Period',
            labels={'hour': 'Hour', 'likes_count': 'Likes Count'}
        )
        fig_html_list.append(fig4.to_html(full_html=False))

    # Chart 4: Relationship between likes and comments (Scatter Plot)
    if {'likes_count', 'comments_count'}.issubset(df.columns):
        fig5 = px.scatter(
            df,
            x='likes_count',
            y='comments_count',
            title='Relationship Between Likes and Comments',
            labels={'likes_count': 'Likes Count', 'comments_count': 'Comments Count'}
        )
        fig_html_list.append(fig5.to_html(full_html=False))

    return fig_html_list

'''

import matplotlib.pyplot as plt

def handle_linkedin_data(df):
    fig_html_list = []

    # Chart 1: Company distribution (Bar Chart)
    if 'Company_Name' in df.columns:
        company_counts = df['Company_Name'].value_counts().reset_index()  # 重置索引
    company_counts.columns = ['Company Name', 'Count']  # 重命名列
    fig1 = px.bar(
        company_counts,
        x='Company Name',  # 使用正确的列名
        y='Count',
        title='Company Distribution',
        labels={'Company Name': 'Company Name', 'Count': 'Count'}
    )
    fig_html_list.append(fig1.to_html(full_html=False))

    # Chart 2: Class distribution (Bar Chart)
    if 'Class' in df.columns:
        class_counts = df['Class'].value_counts().reset_index()
        class_counts.columns = ['Class', 'Count']
        fig2 = px.bar(
            class_counts,
            x='Class',
            y='Count',
            title='Class Distribution',
            labels={'Class': 'Class', 'Count': 'Count'}
        )
        fig_html_list.append(fig2.to_html(full_html=False))

    # Chart 3: Job distribution by location (Pie Chart)
    if 'Location' in df.columns:
        location_counts = df['Location'].value_counts().reset_index()
        location_counts.columns = ['Location', 'Count']
        fig3 = px.pie(
            location_counts,
            names='Location',
            values='Count',
            title='Job Distribution by Location',
        )
        fig_html_list.append(fig3.to_html(full_html=False))

    # Chart 4: Skill demand distribution (Bar Chart)
    skill_columns = [
        'PYTHON', 'C++', 'JAVA', 'HADOOP', 'SCALA', 'FLASK', 'PANDAS',
        'SPARK', 'NUMPY', 'PHP', 'SQL', 'MYSQL', 'CSS', 'MONGODB', 'NLTK',
        'TENSORFLOW', 'LINUX', 'RUBY', 'JAVASCRIPT', 'DJANGO', 'REACT',
        'REACTJS', 'AI', 'UI', 'TABLEAU', 'NODEJS', 'EXCEL', 'POWER BI',
        'SELENIUM', 'HTML', 'ML'
    ]
    if set(skill_columns).intersection(df.columns):
        skill_counts = {skill: df[skill].sum() for skill in skill_columns if skill in df.columns}
        fig4 = px.bar(
            x=list(skill_counts.keys()),
            y=list(skill_counts.values()),
            title='Skill Demand Distribution',
            labels={'x': 'Skill', 'y': 'Demand Count'}
        )
        fig_html_list.append(fig4.to_html(full_html=False))

    # Chart 5: Followers count distribution by company (Bar Chart)
    if 'LinkedIn_Followers' in df.columns and 'Company_Name' in df.columns:
        followers_by_company = df.groupby('Company_Name')['LinkedIn_Followers'].sum().reset_index()
        fig5 = px.bar(
            followers_by_company.sort_values('LinkedIn_Followers', ascending=False),
            x='Company_Name',
            y='LinkedIn_Followers',
            title='Followers Count by Company',
            labels={'Company_Name': 'Company Name', 'LinkedIn_Followers': 'Followers Count'}
        )
        fig_html_list.append(fig5.to_html(full_html=False))

    # Chart 6: Industry distribution (Pie Chart)
    if 'Company_Name' in df.columns and 'LinkedIn_Followers' in df.columns:
        # Group by 'Company_Name' and sum up 'LinkedIn_Followers'
        followers_by_company = df.groupby('Company_Name')['LinkedIn_Followers'].sum()

        # Create horizontal bar chart
        followers_by_company.sort_values(ascending=False).plot(kind='barh', figsize=(10, 6), color='skyblue')

        # Add title and labels
        plt.title('LinkedIn Followers by Company')
        plt.xlabel('LinkedIn Followers')
        plt.ylabel('Company Name')

        # Show the plot
        plt.tight_layout()
        plt.show()

    return fig_html_list





def download_csv(request):
    # Retrieve uploaded data and original file name from the session
    uploaded_data = request.session.get('uploaded_data', None)
    headers = request.session.get('uploaded_headers', None)
    original_file_name = request.session.get('original_file_name', "uploaded_data.csv")  # Default if no name is found
    
    if not uploaded_data or not headers:
        messages.error(request, "No uploaded data available for download. Please upload a file first.")
        return redirect('upload_csv_or_xlsx')
    
    # Ensure the file has a .csv extension
    if not original_file_name.endswith('.csv'):
        original_file_name += '.csv'

    # Create the HTTP response with the correct content type
    response = HttpResponse(content_type='text/csv')
    response['Content-Disposition'] = f'attachment; filename="{original_file_name}"'

    
    # Write uploaded data to CSV
    writer = csv.DictWriter(response, fieldnames=headers)
    writer.writeheader()
    writer.writerows(uploaded_data)

    return response

def marketing_page(request):
    testimonials = Testimonial.objects.all().order_by("-created_at")
    return render(request, 'main/marketing_page.html', {'testimonials': testimonials})

def testimonial_page(request):
    return render(request, 'main/testimonial_page.html')

def login_rate(request):
    return render(request, 'main/rate_to_login.html')

def manage_visibility(request):
    is_authenticated = request.session.get('is_authenticated', False)

    if not is_authenticated:
        return redirect('login')

    username = request.session.get('username')
    user_account = UserAccount.objects.get(username=username)

    # Query DataItem using UserAccount
    data_items = DataItem.objects.filter(businessman=user_account)
    
    print(f"User Account: {user_account}")
    print(f"Data Items: {data_items}")

    return render(request, 'main/manage_visibility.html', {'data_items': data_items})

def update_visibility(request, data_item_id):
    username = request.session.get('username')
    user_account = UserAccount.objects.get(username=username)

    data_item = get_object_or_404(DataItem, id=data_item_id, businessman=user_account)
    
    if request.method == 'POST':
        form = VisibilitySettingsForm(request.POST, instance=data_item)
        if form.is_valid():
            form.save()
            return JsonResponse({'success': True, 'message': 'Visibility settings updated successfully.'})
        else:
            return JsonResponse({'success': False, 'message': 'Failed to update visibility settings. Please retry.'})
    
    form = VisibilitySettingsForm(instance=data_item)
    return render(request, 'main/update_visibility.html', {'form': form, 'data_item': data_item})

def scrape_content_creator(request):
    return scrape_profile(request)

def scrape_data_analyst(request):
    return scrape_profile(request)

def admin_login(request):
    if request.method == 'POST':
        username = request.POST.get('username')
        password = request.POST.get('password')
        
        # Use Django's built-in authentication system
        user = authenticate(request, username=username, password=password)
        if user is not None and user.is_staff:
            login(request, user)
            return redirect('dashboard')
        else:
            messages.error(request, "Invalid credentials or not an admin user!")
    
    return render(request, 'main/login.html')

def submit_testimonial(request):
    if request.method == 'POST':
        name = request.POST.get('name')
        testimonial = request.POST.get('testimonial')
        rating = request.POST.get('rating')

        if name and testimonial and rating:
            Testimonial.objects.create(name=name, testimonial=testimonial, rating=rating)
            messages.success(request, 'Thank you for your testimonial!')
            return redirect('testimonial_page')
        else:
            messages.error(request, 'All fields are required.')

    return render(request, 'main/testimonial_page.html')



from django.shortcuts import render
from django.http import JsonResponse, HttpResponse
import pandas as pd
import networkx as nx
from neo4j import GraphDatabase
import os
import json

def preds(request):
    return render(request, 'main/preds.html')  

import os
from django.views.decorators.csrf import csrf_exempt
from django.http import JsonResponse

@csrf_exempt
def save_csv(request):
    if request.method == 'POST':
        csv_content = request.POST.get('csvContent')
        if csv_content:
            # Determine the next available filename
            directory = os.path.join(settings.BASE_DIR, 'media', 'downloads')
            os.makedirs(directory, exist_ok=True)
            existing_files = os.listdir(directory)
            next_number = len(existing_files) + 1
            filename = f'user{next_number}.csv'
            filepath = os.path.join(directory, filename)

            # Save the CSV content to the file
            with open(filepath, 'w', newline='', encoding='utf-8') as file:
                file.write(csv_content)

            return JsonResponse({'success': True})
        return JsonResponse({'success': False, 'error': 'No CSV content provided'})
    return JsonResponse({'success': False, 'error': 'Invalid request method'})
# -----------------------------------------------------------------------------------

# ------------- the better looking neo4j visualizer -------------------------------
import logging
import json
from django.shortcuts import render
from neo4j import GraphDatabase
import networkx as nx

# Neo4j connection details
NEO4J_URI = "bolt://localhost:7687"
NEO4J_USER = "neo4j"
NEO4J_PASSWORD = "graphbeek"

def graph_view(request):
    driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))

    with driver.session() as session:
        # Fetch nodes and edges together
        result = session.run("MATCH p=()-[r:CO_OCCURS_WITH]->() RETURN p")
        
        nodes = []
        edges = []
        node_ids = set()
        
        for record in result:
            for node in record["p"].nodes:
                if node.id not in node_ids:
                    nodes.append({"id": node.id, "label": node["name"]})
                    node_ids.add(node.id)
            for rel in record["p"].relationships:
                edges.append({"from": rel.start_node.id, "to": rel.end_node.id})

    # Create a NetworkX graph from the nodes and edges for centrality calculations
    G = nx.Graph()
    for node in nodes:
        G.add_node(node['id'])
    for edge in edges:
        G.add_edge(edge['from'], edge['to'])

    # Calculate centrality measures
    betweenness = nx.betweenness_centrality(G)
    try:
        eigenvector = nx.eigenvector_centrality(G, max_iter=1000)
    except nx.PowerIterationFailedConvergence:
        eigenvector = {node: 0 for node in G.nodes}
    degree = nx.degree_centrality(G)
    closeness = nx.closeness_centrality(G)

    # Update nodes with centrality measures
    for node in nodes:
        node['betweenness'] = betweenness[node['id']]
        node['eigenvector'] = eigenvector[node['id']]
        node['degree'] = degree[node['id']]
        node['closeness'] = closeness[node['id']]

    # Ensure all nodes referenced in edges are included in the nodes array
    for edge in edges:
        if edge['from'] not in node_ids:
            nodes.append({"id": edge['from'], "label": str(edge['from']), "betweenness": 0, "eigenvector": 0, "degree": 0, "closeness": 0})
            node_ids.add(edge['from'])
        if edge['to'] not in node_ids:
            nodes.append({"id": edge['to'], "label": str(edge['to']), "betweenness": 0, "eigenvector": 0, "degree": 0, "closeness": 0})
            node_ids.add(edge['to'])

    # Sort nodes by betweenness centrality and get top 10
    top_10_nodes = sorted(nodes, key=lambda x: x['betweenness'], reverse=True)[:10]

    context = {
        "nodes": json.dumps(nodes),  # Serialize nodes to JSON
        "edges": json.dumps(edges),  # Serialize edges to JSON
        "centrality_results": top_10_nodes
    }

    #print(context)  # Debug statement to print the context

    return render(request, 'main/graph.html', context)

import os
from django.http import JsonResponse
from django.shortcuts import render
from django.views.decorators.csrf import csrf_exempt
from django.core.files.storage import default_storage


@csrf_exempt
def save_visualization(request):
    if request.method == 'POST' and request.FILES.get('image'):
        image = request.FILES['image']
        diag_dir = os.path.join('diag')
        if not os.path.exists(diag_dir):
            os.makedirs(diag_dir)
        
        # Determine the next available filename
        existing_files = os.listdir(diag_dir)
        next_number = len(existing_files) + 1
        filename = f'vis{next_number}.png'
        filepath = os.path.join(diag_dir, filename)
        
        with default_storage.open(filepath, 'wb+') as destination:
            for chunk in image.chunks():
                destination.write(chunk)
        
        return JsonResponse({'status': 'success', 'filename': filename})
    return JsonResponse({'status': 'error'}, status=400)



# -------------------- Implementing Machine Learning for Instagram ---------------------------------

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
from datetime import datetime

from sklearn.ensemble import GradientBoostingRegressor

def predict_engagement(request):
    if request.method == 'POST':
        # Load the data
        df = pd.read_csv('C:/Users/Welcome/Desktop/Predictest/my-django-app/josh_edit.csv')

        # Preprocess the data
        df['Is Video'] = df['Is Video'].fillna(False).astype(int)  # Fill NaN and convert boolean to int
        df.dropna(subset=['Likes', 'Comments'], inplace=True)  # Drop rows with missing target values

        # Handle NaN values in features
        df.fillna({
            'Day': 0,
            'Month': 0,
            'Year': 0,
            'Hour': 0,
            'Video Duration': 0
        }, inplace=True)

        # Extract date and hour from the request
        selected_date = request.POST.get('date')
        selected_hour = request.POST.get('hour')

        if not selected_date or selected_hour is None:
            return render(request, 'main/ml_predictions.html', {'error': 'Please provide both date and hour.'})

        selected_hour = int(selected_hour)

        # Convert selected date to day of the week
        date_object = datetime.strptime(selected_date, '%Y-%m-%d')
        day_of_week = date_object.day  # Extract the day from the date

        # Features and target variables
        X = df[['Day', 'Month', 'Year', 'Hour', 'Is Video', 'Video Duration']]
        y_likes = df['Likes']
        y_comments = df['Comments']

        # Normalize the features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        # Split the data
        X_train, X_test, y_train_likes, y_test_likes = train_test_split(X_scaled, y_likes, test_size=0.2, random_state=42)
        _, _, y_train_comments, y_test_comments = train_test_split(X_scaled, y_comments, test_size=0.2, random_state=42)

        # Hyperparameter tuning for Likes using Gradient Boosting Regressor
        param_grid_likes = {
            'n_estimators': [100, 200, 300],
            'learning_rate': [0.01, 0.1, 0.2],
            'max_depth': [3, 5, 7]
        }
        model_likes = GridSearchCV(GradientBoostingRegressor(random_state=42), param_grid_likes, cv=5)
        model_likes.fit(X_train, y_train_likes)

        # Hyperparameter tuning for Comments using Gradient Boosting Regressor
        param_grid_comments = {
            'n_estimators': [100, 200, 300],
            'learning_rate': [0.01, 0.1, 0.2],
            'max_depth': [3, 5, 7]
        }
        
        model_comments = GridSearchCV(GradientBoostingRegressor(random_state=42), param_grid_comments, cv=5)
        model_comments.fit(X_train, y_train_comments)

        # Make predictions
        predictions_likes = model_likes.predict(X_test)
        predictions_comments = model_comments.predict(X_test)

        # Calculate accuracy (mean squared error)
        mse_likes = mean_squared_error(y_test_likes, predictions_likes)
        mse_comments = mean_squared_error(y_test_comments, predictions_comments)

        # Prepare results for rendering
        results = {
            'videos_with_hashtags': {
                'predicted_likes': predictions_likes.tolist(),
                'predicted_comments': predictions_comments.tolist(),
                'mse_likes': mse_likes,
                'mse_comments': mse_comments
            },
            'images_with_hashtags': {
                'predicted_likes': predictions_likes.tolist(),
                'predicted_comments': predictions_comments.tolist(),
                'mse_likes': mse_likes,
                'mse_comments': mse_comments
            },
            'videos_without_hashtags': {
                'predicted_likes': predictions_likes.tolist(),
                'predicted_comments': predictions_comments.tolist(),
                'mse_likes': mse_likes,
                'mse_comments': mse_comments
            },
            'images_without_hashtags': {
                'predicted_likes': predictions_likes.tolist(),
                'predicted_comments': predictions_comments.tolist(),
                'mse_likes': mse_likes,
                'mse_comments': mse_comments
            }
        }

        # Create a list of hours for the template
        hours = list(range(24))

        return render(request, 'main/ml_predictions.html', {'results': results, 'hours': hours})

    # If GET request, render the form
    return render(request, 'main/ml_predictions.html')