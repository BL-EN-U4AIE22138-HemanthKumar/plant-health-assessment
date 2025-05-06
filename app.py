import os
import uuid
import boto3
from botocore.exceptions import NoCredentialsError, ClientError
from flask import Flask, render_template, request, jsonify, redirect, url_for, flash, session
from werkzeug.utils import secure_filename
from utils.image_processing import get_plant_bounding_boxes, get_cropped_patches, classify_plant_health_rules # Using rule-based for now
from model.model_utils import load_prediction_model, predict_patch_health # Use this if you have a model file

# --- Configuration ---
UPLOAD_FOLDER = 'static/uploads/' # Temporary local storage before S3
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}
MAX_CONTENT_LENGTH = 16 * 1024 * 1024 # 16MB Max Upload Size

# --- AWS Configuration (Load from environment variables for security) ---
# Make sure to set these environment variables in your deployment environment
# For local testing, you can use a .env file and python-dotenv, but DO NOT commit .env
AWS_ACCESS_KEY_ID = os.environ.get('AWS_ACCESS_KEY_ID')
AWS_SECRET_ACCESS_KEY = os.environ.get('AWS_SECRET_ACCESS_KEY')
AWS_REGION = os.environ.get('AWS_REGION', 'ap-south-1') # Example region
S3_BUCKET = os.environ.get('S3_BUCKET_NAME')
COGNITO_USER_POOL_ID = os.environ.get('COGNITO_USER_POOL_ID')
COGNITO_APP_CLIENT_ID = os.environ.get('COGNITO_APP_CLIENT_ID')
COGNITO_DOMAIN = os.environ.get('COGNITO_DOMAIN') # e.g., your-domain.auth.us-east-1.amazoncognito.com

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = MAX_CONTENT_LENGTH
app.secret_key = os.environ.get('FLASK_SECRET_KEY') # Change for production!

# --- AWS Clients ---
s3_client = boto3.client(
    's3',
    aws_access_key_id=AWS_ACCESS_KEY_ID,
    aws_secret_access_key=AWS_SECRET_ACCESS_KEY,
    region_name=AWS_REGION
)
cognito_client = boto3.client(
    'cognito-idp',
     aws_access_key_id=AWS_ACCESS_KEY_ID,
    aws_secret_access_key=AWS_SECRET_ACCESS_KEY,
    region_name=AWS_REGION
)

# --- Model Loading (Placeholder - uncomment and adapt if using ML model) ---
model = load_prediction_model('model/plant_health_assesment_RESNET50.h5')
class_labels = ['healthy', 'not_healthy'] # Make sure order matches model training

# --- Helper Functions ---
def allowed_file(filename):
    """Checks if the uploaded file has an allowed extension."""
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def upload_to_s3(file_path, bucket_name, object_name=None):
    """Uploads a file to an S3 bucket."""
    if object_name is None:
        object_name = os.path.basename(file_path)
    try:
        s3_client.upload_file(file_path, bucket_name, object_name)
        # Generate a presigned URL for temporary access (adjust expiration as needed)
        url = s3_client.generate_presigned_url('get_object',
                                               Params={'Bucket': bucket_name, 'Key': object_name},
                                               ExpiresIn=3600) # URL valid for 1 hour
        return url
    except FileNotFoundError:
        print("The file was not found")
        return None
    except NoCredentialsError:
        print("Credentials not available")
        return None
    except ClientError as e:
        print(f"S3 Client Error: {e}")
        return None

def requires_auth(f):
    """Decorator to check if user is logged in."""
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if 'user' not in session:
            flash('Please log in to access this page.', 'warning')
            return redirect(url_for('login'))
        return f(*args, **kwargs)
    return decorated_function


# --- Routes ---
@app.route('/')
def index():
    """Serves the main page (login/signup or dashboard)."""
    if 'user' in session:
         return redirect(url_for('dashboard'))
    # Simple landing/login prompt if not logged in
    return render_template('login.html') # Or a dedicated landing page

@app.route('/login', methods=['GET', 'POST'])
def login():
    """Handles user login."""
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        try:
            response = cognito_client.initiate_auth(
                ClientId=COGNITO_APP_CLIENT_ID,
                AuthFlow='USER_PASSWORD_AUTH',
                AuthParameters={
                    'USERNAME': username,
                    'PASSWORD': password
                }
            )
            # Login successful - Store user info in session
            # IMPORTANT: In a real app, handle MFA, new password challenges etc.
            # Store access token securely if needed for API calls
            session['user'] = username
            # You might want to fetch user attributes here as well
            session['access_token'] = response['AuthenticationResult']['AccessToken']
            flash('Login successful!', 'success')
            return redirect(url_for('dashboard'))
        except cognito_client.exceptions.NotAuthorizedException:
            flash('Incorrect username or password.', 'danger')
        except cognito_client.exceptions.UserNotFoundException:
            flash('User does not exist.', 'danger')
        except Exception as e:
            flash(f'An error occurred during login: {e}', 'danger')

    return render_template('login.html')

@app.route('/signup', methods=['GET', 'POST'])
def signup():
    """Handles user signup."""
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        email = request.form['email'] # Assuming email is collected

        try:
            response = cognito_client.sign_up(
                ClientId=COGNITO_APP_CLIENT_ID,
                Username=username,
                Password=password,
                UserAttributes=[
                    {'Name': 'email', 'Value': email},
                    # Add other required attributes here
                ]
            )
            flash('Signup successful! Please check your email to confirm your account.', 'success')
            # Redirect to login or a confirmation page
            return redirect(url_for('login'))
        except cognito_client.exceptions.UsernameExistsException:
            flash('Username already exists.', 'danger')
        except cognito_client.exceptions.InvalidPasswordException as e:
             flash(f'Invalid password: {e}', 'danger') # Provide specifics if possible
        except cognito_client.exceptions.InvalidParameterException as e:
            flash(f'Invalid parameter: {e}', 'danger') # e.g., invalid email format
        except Exception as e:
            flash(f'An error occurred during signup: {e}', 'danger')

    return render_template('signup.html')

@app.route('/logout')
def logout():
    """Logs the user out."""
    session.pop('user', None)
    # session.pop('access_token', None) # Clear token if stored
    flash('You have been logged out.', 'info')
    # Invalidate Cognito tokens if necessary (more complex, involves refresh tokens)
    return redirect(url_for('login'))


@app.route('/dashboard')
@requires_auth # Protect this route
def dashboard():
    """Serves the main dashboard page."""
    return render_template('dashboard.html', username=session.get('user'))

@app.route('/upload', methods=['POST'])
@requires_auth
def upload_file():
    """Handles file upload, saves to S3, and returns S3 URL."""
    if 'plantImage' not in request.files:
        return jsonify({'error': 'No file part'}), 400
    file = request.files['plantImage']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        # Create a unique filename to avoid S3 conflicts
        unique_filename = f"{uuid.uuid4()}_{filename}"
        # Save locally temporarily
        local_path = os.path.join(app.config['UPLOAD_FOLDER'], unique_filename)
        os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True) # Ensure folder exists
        try:
            file.save(local_path)

            # Upload to S3
            s3_url = upload_to_s3(local_path, S3_BUCKET, unique_filename)

            # Clean up local file after successful S3 upload
            if s3_url:
                 if os.path.exists(local_path):
                     os.remove(local_path)
                 return jsonify({'s3_url': s3_url, 'filename': unique_filename}) # Return S3 URL
            else:
                 # Optionally remove local file even if S3 upload fails
                 if os.path.exists(local_path):
                     os.remove(local_path)
                 return jsonify({'error': 'Failed to upload to S3'}), 500

        except Exception as e:
            # Clean up local file on error
            if os.path.exists(local_path):
                os.remove(local_path)
            return jsonify({'error': f'Error saving or uploading file: {str(e)}'}), 500
        finally:
             # Ensure cleanup happens even if unexpected errors occur before explicit removal
             if os.path.exists(local_path):
                 try:
                     os.remove(local_path)
                 except OSError as e:
                     print(f"Error removing temporary file {local_path}: {e}")


    return jsonify({'error': 'File type not allowed'}), 400


@app.route('/process', methods=['POST'])
@requires_auth
def process_image():
    """Processes the uploaded image: gets boxes, crops, predicts."""
    data = request.get_json()
    s3_key = data.get('filename') # Use the unique filename (S3 key)
    if not s3_key or not S3_BUCKET:
        return jsonify({'error': 'Missing filename or S3 bucket configuration'}), 400

    # --- Download image temporarily from S3 for processing ---
    # Note: For large images or frequent processing, it's better to
    # trigger a separate processing job (e.g., Lambda) on S3 upload.
    local_download_path = os.path.join(app.config['UPLOAD_FOLDER'], s3_key)
    os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
    results = []
    try:
        s3_client.download_file(S3_BUCKET, s3_key, local_download_path)

        # 1. Get Bounding Boxes
        boxes = get_plant_bounding_boxes(local_download_path, show=False) # Don't show plots

        if not boxes:
             return jsonify({'boxes': [], 'results': [], 'error': 'No plants detected.'})

        # 2. Get Cropped Patches
        cropped_patches_data = get_cropped_patches(local_download_path, boxes, show=False) # Get image data

        # 3. Classify Health (Using rule-based for now)
        # Replace with model prediction if model is ready
        health_results = classify_plant_health_rules(cropped_patches_data)

        # --- Prepare results for frontend ---
        # Convert numpy arrays (cropped images) to base64 strings or save them
        # to S3 and return URLs. Base64 is simpler for fewer/smaller images.
        # Saving to S3 is better for many/large images. Let's use S3.

        patch_s3_urls = []
        for i, patch_img in enumerate(cropped_patches_data):
            patch_filename = f"patch_{i}_{s3_key}"
            patch_local_path = os.path.join(app.config['UPLOAD_FOLDER'], patch_filename)
            # Save patch locally first
            import cv2
            cv2.imwrite(patch_local_path, patch_img)
            # Upload patch to S3
            patch_url = upload_to_s3(patch_local_path, S3_BUCKET, f"patches/{patch_filename}")
            if patch_url:
                patch_s3_urls.append(patch_url)
            # Clean up local patch file
            if os.path.exists(patch_local_path):
                os.remove(patch_local_path)

        # Combine results
        results = []
        for i, (box_coords) in enumerate(boxes):
             patch_url = patch_s3_urls[i] if i < len(patch_s3_urls) else None
             health_label = health_results[i][1] if i < len(health_results) else 'Error'
             results.append({
                 'box': box_coords,
                 'patch_url': patch_url,
                 'health': health_label
             })

        return jsonify({'results': results})

    except FileNotFoundError:
         return jsonify({'error': 'Image not found for processing (check S3 download)'}), 404
    except Exception as e:
        print(f"Processing Error: {e}") # Log the error server-side
        import traceback
        traceback.print_exc()
        return jsonify({'error': f'Error during processing: {str(e)}'}), 500
    finally:
        # Clean up downloaded file
        if os.path.exists(local_download_path):
            try:
                os.remove(local_download_path)
            except OSError as e:
                 print(f"Error removing downloaded file {local_download_path}: {e}")


if __name__ == '__main__':
    # Ensure the upload folder exists
    os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
    # Use debug=True only for development
    app.run(debug=True, host='0.0.0.0', port=5000) # Run on 0.0.0.0 to be accessible on network/EC2
