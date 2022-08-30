from bs4 import BeautifulSoup
import urllib.request


queries = [
	"Shilpa Sankpal Dermacos Skin Hair & Laser Clinic Navi Mumbai Kharghar",
	"Prateek Sondhi Derma Cells Delhi"
]

for query in queries:
	url = 'https://google.com/search?q='+"+".join(q for q in query.split(" "))
	print(url)

	# Perform the request
	request = urllib.request.Request(url)

	# Set a normal User Agent header, otherwise Google will block the request.
	request.add_header('User-Agent', 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_6) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/87.0.4280.88 Safari/537.36')
	raw_response = urllib.request.urlopen(request).read()

	# Read the repsonse as a utf-8 string
	html = raw_response.decode("utf-8")


	# The code to get the html contents here.

	soup = BeautifulSoup(html, 'html.parser')

	# Find all the search result divs
	divs = soup.find_all("span", class_="Aq14fc")

	# print(divs)
	for div in divs:
		# Search for a h3 tag
		print(div.get_text())