/**
 * Frontend JavaScript for Plant Health Assessment App
 */

// --- DOM Elements ---
const uploadForm = document.getElementById('upload-form');
const imageInput = document.getElementById('plant-image-input');
const imageContainer = document.getElementById('image-container');
const uploadedImage = document.getElementById('uploaded-image');
const patchesContainer = document.getElementById('patches-container');
const resultsArea = document.getElementById('results-area');
const loader = document.getElementById('loader');
const errorMessageDiv = document.getElementById('error-message');
const clearButton = document.getElementById('clear-button');

// --- Global Variables ---
let currentFilename = null; // To store the unique filename from S3 upload

// --- Event Listeners ---
if (uploadForm) {
    uploadForm.addEventListener('submit', handleUploadAndAnalyze);
}

if (clearButton) {
    clearButton.addEventListener('click', clearResults);
}

// --- Functions ---

/**
 * Handles the form submission for uploading and analyzing the image.
 * @param {Event} event - The form submission event.
 */
async function handleUploadAndAnalyze(event) {
    event.preventDefault(); // Prevent default form submission
    clearResults(); // Clear previous results
    showLoader(true);
    showError(null); // Clear previous errors

    const formData = new FormData(uploadForm);
    const file = imageInput.files[0];

    if (!file) {
        showError('Please select an image file.');
        showLoader(false);
        return;
    }

    // Basic client-side validation (optional, as backend also validates)
    const allowedTypes = ['image/png', 'image/jpeg', 'image/jpg'];
    if (!allowedTypes.includes(file.type)) {
         showError('Invalid file type. Please upload PNG, JPG, or JPEG.');
         showLoader(false);
         return;
    }
     // Optional: Client-side size check (backend also checks)
     const maxSize = 16 * 1024 * 1024; // 16MB
     if (file.size > maxSize) {
         showError('File is too large. Maximum size is 16MB.');
         showLoader(false);
         return;
     }


    try {
        // 1. Upload the image to the backend (which uploads to S3)
        const uploadResponse = await fetch('/upload', {
            method: 'POST',
            body: formData,
        });

        if (!uploadResponse.ok) {
            const errorData = await uploadResponse.json();
            throw new Error(errorData.error || `Upload failed with status: ${uploadResponse.status}`);
        }

        const uploadData = await uploadResponse.json();
        const s3Url = uploadData.s3_url;
        currentFilename = uploadData.filename; // Store the unique filename (S3 key)

        if (!s3Url || !currentFilename) {
             throw new Error('Upload successful, but did not receive necessary data (URL/Filename).');
        }

        // Display the uploaded image from S3 URL
        uploadedImage.src = s3Url;
        uploadedImage.alt = "Uploaded Plant Image";
        resultsArea.style.display = 'block'; // Show results area container

        // 2. Send request to process the image (using the filename/S3 key)
        const processResponse = await fetch('/process', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({ filename: currentFilename }),
        });

        if (!processResponse.ok) {
            const errorData = await processResponse.json();
            throw new Error(errorData.error || `Processing failed with status: ${processResponse.status}`);
        }

        const processData = await processResponse.json();

        // 3. Display the results
        displayResults(processData.results, s3Url);

    } catch (error) {
        console.error('Error:', error);
        showError(`An error occurred: ${error.message}`);
    } finally {
        showLoader(false); // Hide loader regardless of success or failure
    }
}

/**
 * Displays the analysis results (bounding boxes, patches, health status).
 * @param {Array} results - Array of result objects from the backend.
 * @param {string} originalImageUrl - The URL of the originally uploaded image.
 */
function displayResults(results, originalImageUrl) {
    // Clear previous boxes and patches
    clearBoundingBoxes();
    patchesContainer.innerHTML = '';

    if (!results || results.length === 0) {
        // Display a message if no plants/results were found
        const noResultMsg = document.createElement('p');
        noResultMsg.textContent = 'No plants detected or analysis failed.';
        noResultMsg.className = 'text-gray-500 italic';
        patchesContainer.appendChild(noResultMsg);
        // Keep the original image visible
        uploadedImage.src = originalImageUrl;
        imageContainer.style.display = 'inline-block';
        resultsArea.style.display = 'block';
        return;
    }


    // Ensure the original image is loaded before drawing boxes
    uploadedImage.onload = () => {
        results.forEach((result, index) => {
            // Draw Bounding Box
            if (result.box) {
                drawBoundingBox(result.box);
            }

            // Display Cropped Patch and Health Status
            if (result.patch_url) {
                const patchDiv = document.createElement('div');
                patchDiv.className = 'border rounded-lg overflow-hidden shadow-sm p-2 text-center bg-gray-50';

                const patchImg = document.createElement('img');
                patchImg.src = result.patch_url;
                patchImg.alt = `Patch ${index + 1}`;
                patchImg.className = 'w-full h-auto object-cover mb-2 rounded';
                // Add error handling for patch images
                patchImg.onerror = () => {
                    patchImg.alt = `Patch ${index + 1} (Load Error)`;
                    patchImg.src = 'https://placehold.co/100x100/cccccc/ffffff?text=Error'; // Placeholder
                };


                const healthStatus = document.createElement('p');
                healthStatus.textContent = `Status: ${result.health || 'N/A'}`;
                healthStatus.className = `text-sm font-semibold ${result.health === 'healthy' ? 'text-green-600' : 'text-red-600'}`;

                patchDiv.appendChild(patchImg);
                patchDiv.appendChild(healthStatus);
                patchesContainer.appendChild(patchDiv);
            }
        });
        // Ensure image container is visible after processing boxes/patches
         imageContainer.style.display = 'inline-block';
    };
     // Set src again in case it was already cached and onload didn't fire initially
     if (uploadedImage.complete) {
         uploadedImage.onload(); // Fire manually if already loaded
     }
     uploadedImage.src = originalImageUrl; // Ensure src is set


    resultsArea.style.display = 'block';
}

/**
 * Draws a single bounding box overlay on the image container.
 * @param {Array} box - Coordinates [x1, y1, x2, y2].
 */
function drawBoundingBox(box) {
    const [x1, y1, x2, y2] = box;

    // Get image dimensions AFTER it has loaded
    const imgWidth = uploadedImage.naturalWidth;
    const imgHeight = uploadedImage.naturalHeight;

    // Get display dimensions of the image element
    const displayWidth = uploadedImage.offsetWidth;
    const displayHeight = uploadedImage.offsetHeight;

    if (imgWidth === 0 || imgHeight === 0 || displayWidth === 0 || displayHeight === 0) {
        console.warn("Cannot draw bounding box: Image dimensions not available yet.");
        return; // Don't draw if dimensions aren't ready
    }


    // Calculate scaling factors
    const widthScale = displayWidth / imgWidth;
    const heightScale = displayHeight / imgHeight;

    const boxDiv = document.createElement('div');
    boxDiv.className = 'bounding-box';

    // Apply scaling to box coordinates and dimensions
    boxDiv.style.left = `${x1 * widthScale}px`;
    boxDiv.style.top = `${y1 * heightScale}px`;
    boxDiv.style.width = `${(x2 - x1) * widthScale}px`;
    boxDiv.style.height = `${(y2 - y1) * heightScale}px`;

    imageContainer.appendChild(boxDiv);
}

/**
 * Clears all drawn bounding boxes.
 */
function clearBoundingBoxes() {
    const existingBoxes = imageContainer.querySelectorAll('.bounding-box');
    existingBoxes.forEach(box => box.remove());
}

/**
 * Clears all results and resets the form/display.
 */
function clearResults() {
    showLoader(false);
    showError(null);
    resultsArea.style.display = 'none';
    patchesContainer.innerHTML = '';
    clearBoundingBoxes();
    uploadedImage.src = '#'; // Clear image preview
    uploadedImage.alt = '';
    imageContainer.style.display = 'none'; // Hide container
    uploadForm.reset(); // Reset the file input
    currentFilename = null;
}

/**
 * Shows or hides the loader animation.
 * @param {boolean} show - True to show, false to hide.
 */
function showLoader(show) {
    if (loader) {
        loader.style.display = show ? 'block' : 'none';
    }
}

/**
 * Displays an error message or clears it.
 * @param {string|null} message - The error message string, or null to clear.
 */
function showError(message) {
     if (errorMessageDiv) {
        if (message) {
            errorMessageDiv.textContent = message;
            errorMessageDiv.style.display = 'block';
        } else {
            errorMessageDiv.textContent = '';
            errorMessageDiv.style.display = 'none';
        }
    }
}

// --- Initial Setup ---
// Ensure results area is hidden initially
document.addEventListener('DOMContentLoaded', () => {
    if(resultsArea) resultsArea.style.display = 'none';
    if(imageContainer) imageContainer.style.display = 'none';
});
