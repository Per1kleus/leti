"""Getting Leti running on a machine that has nothing installed on it.

Two front doors - Leti.exe and "Launch Leti (Windows).bat" - and one thing behind
them: launcher/bootstrap.py, which finds or fetches a Python, installs whatever
requirements.txt asks for that is not already there, and starts the same main.py
a developer runs from the repository.

Nothing in here is imported by the application. Leti does not know it was
launched rather than run, which is the point: there is one Leti, reached two ways.
"""
