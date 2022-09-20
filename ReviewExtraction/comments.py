from selenium import webdriver
from webdriver_manager.chrome import ChromeDriverManager
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By


options = Options()
options.headless = True
# As there are possibilities of different chrome
# browser and we are not sure under which it get
# executed let us use the below syntax
driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()), options=options)

driver.get("http://www.python.org")
elem = driver.find_element(By.NAME, "q")
print(elem)