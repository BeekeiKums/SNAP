from django.urls import path
from . import views, neoviews
from .views import save_csv
from .views import graph_view

urlpatterns = [
    path('marketing_page/', views.marketing_page, name='marketing_page'),  # Updated URL pattern
    path('login/', views.login, name='login'),
    path('logout/', views.logout, name='logout'),  # Ensure this is correctly defined
    
    # Dashboard
    path('dashboard/', views.dashboard, name='dashboard'),
    path('businessman_dashboard/', views.businessman_dashboard, name='businessman_dashboard'),
    path('content-creator-dashboard/', views.content_creator_dashboard, name='content_creator_dashboard'),
    path('data-analyst-dashboard/', views.data_analyst_dashboard, name='data_analyst_dashboard'),

    path('testimonial_page/', views.testimonial_page, name='testimonial_page'),
    path('create_businessman_profile/', views.create_businessman_account, name='create_profile'),
    path('view_profile_businessman/', views.view_profile_content_creator, name='view_profile_businessman'),
    
    # Create account
    path('create_user_account/', views.create_user_account, name='create_user_account'),
    path('create_businessman_account/', views.create_businessman_account, name='create_businessman_account'),
    path('create_content_creator_account/', views.create_content_creator_account, name='create_content_creator_account'),
    path('create_data_analyst_account/', views.create_data_analyst_account, name='create_data_analyst_account'),
    
    # View user accounts
    path('view_user_accounts/', views.view_user_accounts, name='view_user_accounts'),
    path('view_businessman_accounts/', views.view_businessman_accounts, name='view_businessman_accounts'),
    path('view_content_creator_accounts/', views.view_content_creator_accounts, name='view_content_creator_accounts'),
    path('view_data_analyst_accounts/', views.view_data_analyst_accounts, name='view_data_analyst_accounts'),
    
    # Categories
    path('categories/', views.category_list, name='category_list'),
    path('create_category/', views.create_category, name='create_category'),
    path('category/update/<int:category_id>/', views.update_category, name='update_category'),
    
    # Update user account
    path('update_user_account/<int:user_id>/', views.update_user_account, name='update_user_account'),
    
    # Preprocess
    path('auto_preprocess/', views.auto_preprocess, name='auto_preprocess'),
    path('manual_preprocess/', views.manual_preprocess, name='manual_preprocess'),
    
    # CSV
    path('download_csv/', views.download_csv, name='download_csv'),
    path('upload_csv_or_xlsx/', views.upload_csv_or_xlsx, name='upload_csv_or_xlsx'),
    
    # Visualization
    path('create_or_view_visualization/', views.create_or_view_visualization, name='create_or_view_visualization'),
    path('dash/', views.dashboard_view, name='dash'),  # Corrected view function
    
    # Predictive models
    path('test_predictive_models/', views.test_predictive_models, name='test_predictive_models'),
    
    # Scrape
    path('scrape_profile/', views.scrape_profile, name='scrape_profile'),
    path('login_instagram/', views.login_instagram, name='login_instagram'),
    path('scrape_content_creator/', views.scrape_content_creator, name='scrape_content_creator'),
    path('scrape_data_analyst/', views.scrape_data_analyst, name='scrape_data_analyst'),
    
    # Profile
    path('create_profile/', views.create_profile, name='create_profile'),
    path('view_profile/', views.view_profile, name='view_profile'),
    path('update_profile/<str:profile_id>/', views.update_profile, name='update_profile'),
    
    # Testimonial
    path('testimonial_page/', views.testimonial_page, name='testimonial_page'),
    
    # Login rate
    path('login_rate/', views.login_rate, name='login_rate'),
    
    # Visibility
    path('visibility/', views.manage_visibility, name='manage_visibility'),
    path('visibility/<int:data_item_id>/', views.update_visibility, name='update_visibility'),
    
    # Charts
    path('upload_and_view_charts/', views.upload_and_view_charts, name='upload_and_view_charts'),

    # To add and remove rows and columns
    path('save_csv/', views.save_csv, name='save_csv'),
    path('preds/', views.preds, name='preds'),

    # To view the correct neo4j graph
    path('graph_view/', views.graph_view, name='graph_view'),
    path('save-visualization/', views.save_visualization, name='save_visualization'),
    path('upload_and_view_charts/', views.upload_and_view_charts, name='upload_and_view_charts'),  # Add the new URL pattern
    
    


    # To view the Machine Learning Algo of Social Media
    path('predict_engagement/', views.predict_engagement, name='predict_engagement'),

    # Admin login
    path('admin_login/', views.admin_login, name='admin_login'),
]