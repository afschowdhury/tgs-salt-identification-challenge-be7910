1. Environment Setup
Purpose: Extracts custom library files and configures Python path
Key Tasks:
•	Unzips competition-supplied flamingo.zip
•	Moves lib/ folder to working directory
•	Adds library path to Python's module search path


2. Model Initialization
Purpose: Extracts custom library files and configures Python path
Key Tasks:
•	Unzips competition-supplied flamingo.zip
•	Moves lib/ folder to working directory
•	Adds library path to Python's module search path


3. Test Data Preparation
Data Flow:
1.	Collects all .png files from test directory
2.	Creates sorted list of image paths
3.	Implements custom Dataset class for:
	Image normalization (/255.0)
	Channel dimension addition
	ID extraction from filenames


4. Data Loading
Batch size: 32 (optimized for GPU memory)
No shuffling: Maintains test order consistency
Parallel loading: Uses PyTorch's native multiprocessing


5. Prediction Generation
Process:
•	Converts model outputs to probabilities with sigmoid
•	Applies threshold (0.45) to create binary masks
•	Removes singleton dimensions with np.squeeze
Key Considerations:
•	Threshold chosen based on validation performance
•	Batch processing for memory efficiency


6. Run-Length Encoding (RLE)
Kaggle Requirements:
•	Flattening Order: Column-wise (Fortran-style)
•	Format: Space-separated pairs start length
•	Empty Masks: Returns empty string


7. Submission File Creation
Output Specifications:
•	id: Test image ID (e.g., "a8d8a7f3f7")
•	rle_mask: Encoded pixel positions
•	Compression: GZIP for reduced file size
•	File Path: /kaggle/working/submission.csv.gz
