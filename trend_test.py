from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
import os
import time

# Set up download directory to user's root folder
root_dir = os.path.expanduser('~')  # Gets user's home directory
download_dir = root_dir  # Change this if you want a specific subdirectory

# Chrome options configuration
chrome_options = Options()
prefs = {
    "download.default_directory": download_dir,
    "download.prompt_for_download": False,
    "download.directory_upgrade": True,
    "safebrowsing.enabled": True
}
chrome_options.add_experimental_option("prefs", prefs)
chrome_options.add_argument("--disable-infobars")

# Initialize WebDriver
service = Service('D:\YouToob\shorty\environments\chromedriver-win64\chromedriver.exe')
driver = webdriver.Chrome(service=service, options=chrome_options)

try:
    # Navigate to trending page
    driver.get("https://trends.google.com/trending?geo=US&hl=en-US&hours=168")
    
    # Handle cookie consent
    try:
        cookie_btn = WebDriverWait(driver, 10).until(
            EC.element_to_be_clickable((By.XPATH, '//button[contains(., "Accept all")]'))
        )
        cookie_btn.click()
    except:
        print("No cookie consent found or already accepted")
    
    # Click export button
    # Find any tag (div, span, button, etc.) with text "Export"
    export_btn = WebDriverWait(driver, 15).until(
        EC.element_to_be_clickable((By.XPATH, '//*[contains(text(), "Export")]'))
    )
    export_btn.click()

    
    # Click CSV download option
    csv_btn = WebDriverWait(driver, 10).until(
        EC.element_to_be_clickable((By.XPATH, '//div[contains(text(), "Download CSV")]'))
    )
    csv_btn.click()
    
    # Wait for download to complete
    time.sleep(5)  # Adjust based on your internet speed

finally:
    driver.quit()

print(f"CSV file downloaded to: {download_dir}")