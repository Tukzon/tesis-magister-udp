from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.firefox.options import Options
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC

options = Options()
options.add_argument("--headless")
options.binary_location = r'C:\Program Files\Mozilla Firefox\firefox.exe'

driver = webdriver.Remote(command_executor='http://127.0.0.1:4444', options=options)

print("*" * 40)
print("Diario Financiero")
print("*" * 40)

url = "https://www.df.cl/noticias/site/tax/port/all/taxport_3_20__1.html"
print("Scrapping DF - Bolsa & Monedas")
print("Source: "+url)

bypass_url = f"https://12ft.io/{url}"
print("Proxy: "+bypass_url)
print("-" * 40)

driver.get(bypass_url)

WebDriverWait(driver, 10).until(EC.frame_to_be_available_and_switch_to_it((By.ID, "proxy-frame")))

WebDriverWait(driver, 10).until(EC.presence_of_element_located((By.CLASS_NAME, 'col-md-9')))

articles = driver.find_elements(By.CLASS_NAME, 'col-md-9')

for article in articles:
    title = article.find_element(By.TAG_NAME, 'h2').text
    link = article.find_element(By.TAG_NAME, 'a').get_attribute('href')
    print(f'Título: {title}')
    print(f'URL: {link}')
    print('-' * 40)

driver.quit()