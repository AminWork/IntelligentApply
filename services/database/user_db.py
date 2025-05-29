import pymongo

# It's a good practice to get MongoDB connection details from environment variables or a config file
# For this example, I'm hardcoding it, but you should change this in a real application.
MONGO_URI = "mongodb://localhost:27017/"
DB_NAME = "academic_apply_ai"
USER_COLLECTION = "users"

CLIENT = None
DB = None
USER_C = None


def get_mongo_client():
    """Initializes and returns the MongoDB client."""
    global CLIENT
    if CLIENT is None:
        try:
            CLIENT = pymongo.MongoClient(MONGO_URI)
            CLIENT.admin.command('ping') # Verify connection
            print("MongoDB connection successful.")
        except pymongo.errors.ConnectionFailure as e:
            print(f"MongoDB connection failed: {e}")
            CLIENT = None # Reset client on failure
            raise
    return CLIENT

def get_database():
    """Returns the application's database."""
    global DB
    if DB is None:
        client = get_mongo_client()
        if client:
            DB = client[DB_NAME]
    return DB

def get_user_collection():
    """Returns the user collection."""
    global USER_C
    if USER_C is None:
        db = get_database()
        if db:
            USER_C = db[USER_COLLECTION]
    return USER_C

# Define the user fields based on the provided image
USER_FIELDS = [
    "user_cv_extracted_name",
    "user_cv_extracted_email", # Primary key for identifying users for updates
    "user_cv_extracted_phone",
    "user_cv_extracted_address",
    "user_cv_extracted_website_portfolio",
    "user_cv_extracted_linkedin_profile",
    "user_cv_extracted_education_highlights",
    "user_cv_extracted_research_experience_highlights", # Combined from Research and Work Experience
    "user_cv_extracted_work_experience_highlights",
    "user_cv_extracted_publications",
    "user_cv_extracted_skills",
    "user_cv_extracted_research_interests",
    "user_cv_extracted_awards_honors",
    "user_cv_extracted_references_availability", # "References" field
    "user_cv_extracted_summary",
    "cv_latex_path" # Added for storing path to the raw LaTeX CV
]

def add_user(user_data: dict) -> dict:
    """
    Adds a new user to the MongoDB database.

    Args:
        user_data: A dictionary containing user information. 
                   Keys should correspond to USER_FIELDS.

    Returns:
        A dictionary with the status of the operation and the inserted_id if successful.
    """
    try:
        users_collection = get_user_collection()
        if not users_collection:
            return {"status": "error", "message": "Failed to get user collection."}

        # Ensure email is present as it might be used as an identifier
        if not user_data.get("user_cv_extracted_email"):
            return {"status": "error", "message": "User email is required to add a user."}

        # Check if user with this email already exists
        if users_collection.find_one({"user_cv_extracted_email": user_data["user_cv_extracted_email"]}):
            return {"status": "error", "message": f"User with email {user_data['user_cv_extracted_email']} already exists."}

        new_user = {}
        for field in USER_FIELDS:
            new_user[field] = user_data.get(field, "") # Store empty string if field is not provided

        result = users_collection.insert_one(new_user)
        return {
            "status": "success", 
            "message": "User added successfully.", 
            "inserted_id": str(result.inserted_id)
        }
    except pymongo.errors.ConnectionFailure as e:
        return {"status": "error", "message": f"Database connection failed: {e}"}
    except Exception as e:
        return {"status": "error", "message": f"An error occurred while adding user: {e}"}

def update_user(user_email: str, update_data: dict) -> dict:
    """
    Updates an existing user's information in the MongoDB database.

    Args:
        user_email: The email of the user to update.
        update_data: A dictionary containing the fields to update and their new values.
                     Keys should correspond to USER_FIELDS.

    Returns:
        A dictionary with the status of the operation.
    """
    try:
        users_collection = get_user_collection()
        if not users_collection:
            return {"status": "error", "message": "Failed to get user collection."}

        # Construct the update document, only including fields present in USER_FIELDS
        update_doc = {}
        for field, value in update_data.items():
            if field in USER_FIELDS:
                update_doc[field] = value
            # else: you might want to log or handle fields not in USER_FIELDS

        if not update_doc: # No valid fields to update
            return {"status": "error", "message": "No valid fields provided for update."}

        result = users_collection.update_one(
            {"user_cv_extracted_email": user_email},
            {"$set": update_doc}
        )

        if result.matched_count == 0:
            return {"status": "error", "message": f"User with email {user_email} not found."}
        elif result.modified_count == 0:
            return {"status": "success", "message": "No changes made to the user (data might be the same)."}
        else:
            return {"status": "success", "message": "User updated successfully."}
            
    except pymongo.errors.ConnectionFailure as e:
        return {"status": "error", "message": f"Database connection failed: {e}"}
    except Exception as e:
        return {"status": "error", "message": f"An error occurred while updating user: {e}"}

# Example usage (you can remove this or move to a test file)
if __name__ == '__main__':
    # Ensure MongoDB is running and accessible
    # You might need to run `docker-compose up -d` if using the provided docker-compose.yml
    
    # Test adding a user
    sample_user_data = {
        "user_cv_extracted_name": "Dr. Jane Doe",
        "user_cv_extracted_email": "jane.doe@example.com",
        "user_cv_extracted_phone": "123-456-7890",
        "user_cv_extracted_skills": "Python, Machine Learning, Data Analysis",
        "cv_latex_path": "/path/to/cv/jane_doe.tex"
        # Add other fields as needed, or leave them to be empty strings
    }
    
    add_result = add_user(sample_user_data)
    print(f"Add user result: {add_result}")

    # Test adding the same user again (should fail)
    # add_result_fail = add_user(sample_user_data)
    # print(f"Add user (fail) result: {add_result_fail}")

    # To test update, you'd first add a user, then call update_user
    if add_result.get("status") == "success": # only attempt update if add was successful
        updated_data = {
            "user_cv_extracted_phone": "987-654-3210",
            "user_cv_extracted_linkedin_profile": "linkedin.com/in/janedoeupdated",
            "user_cv_extracted_non_existent_field": "this should not be added"
        }
        update_result = update_user("jane.doe@example.com", updated_data)
        print(f"Update user result: {update_result}")

        # Test updating a non-existent user
        # update_non_existent_result = update_user("nonexistent@example.com", updated_data)
        # print(f"Update non-existent user result: {update_non_existent_result}") 