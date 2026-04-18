mkexam
======

Generate exam questions from PDFs, videos, and web pages using Gemini AI.


REQUIREMENTS
------------
- Windows 10 or 11
- Python 3.11 or later  https://www.python.org/downloads/
  (during installation, check "Add python.exe to PATH")
- A Gemini API key  https://aistudio.google.com/apikey
  (free tier is sufficient)


FIRST RUN
---------
1. Open the mkexam folder
2. Open the file named ".env" in Notepad
3. Replace "your_api_key_here" with your Gemini API key and save
4. Double-click run.bat
5. Confirm package installation when prompted
6. Your browser will open automatically at http://localhost:5000


SUBSEQUENT RUNS
---------------
Double-click run.bat — the browser opens automatically.


CUSTOM PORT
-----------
If port 5000 is already in use, run from the command prompt:

    run.bat 8080

Replace 8080 with any available port number.


DATA
----
All decks and cards are stored in the "data\" folder inside the mkexam
directory. To back up your data, copy that folder. To move mkexam to
another computer, copy the entire mkexam folder including "data\".


STOPPING THE SERVER
-------------------
Close the command prompt window that opened when you ran run.bat,
or press Ctrl+C inside it.
