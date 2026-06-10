import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from datetime import datetime
import os
from dotenv import load_dotenv

class EmailNotifier:
    def __init__(self):
        load_dotenv()

        self.sender_email = os.getenv('GMAIL_USER')
        self.app_password = os.getenv('GMAIL_APP_PASSWORD')

        if not self.sender_email or not self.app_password:
            raise ValueError("Email credentials not found in environment variables")

        self.receiver_email = "dledbetter456@gmail.com"

    def send_notification(self, subject, message):
        try:

            msg = MIMEMultipart()
            msg['From'] = self.sender_email
            msg['To'] = self.receiver_email
            msg['Subject'] = subject

            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            full_message = f"Time: {timestamp}\n\n{message}"

            msg.attach(MIMEText(full_message, 'plain'))

            server = smtplib.SMTP('smtp.gmail.com', 587)
            server.starttls()
            server.login(self.sender_email, self.app_password)

            server.send_message(msg)
            server.quit()
            print(f"Email notification sent to {self.receiver_email}")

        except Exception as e:
            print(f"Failed to send email: {str(e)}")

def test_email_setup():
    try:
        notifier = EmailNotifier()
        notifier.send_notification(
            subject="Email Setup Test",
            message="If you receive this, the email notification system is working!"
        )
        print("Email test successful!")
    except Exception as e:
        print(f"Email setup test failed: {str(e)}")

if __name__ == "__main__":
    test_email_setup()
