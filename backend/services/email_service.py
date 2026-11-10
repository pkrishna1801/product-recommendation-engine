import html
import logging
import smtplib
import ssl
import threading
from email.message import EmailMessage
from config import config

logger = logging.getLogger(__name__)


class EmailService:
    """
    Sends transactional email over SMTP (works with Gmail app passwords,
    SendGrid, Mailgun, Brevo, Resend, etc. via their SMTP settings).

    Configured entirely through environment variables; if SMTP_HOST is not
    set, sending is skipped and a log line is written instead.
    """

    def __init__(self):
        self.host = config.get('SMTP_HOST')
        self.port = config.get('SMTP_PORT', 587)
        self.username = config.get('SMTP_USERNAME')
        self.password = config.get('SMTP_PASSWORD')
        self.sender = config.get('MAIL_FROM') or self.username
        self.app_url = config.get('APP_URL')

    @property
    def enabled(self):
        return bool(self.host and self.sender)

    def send(self, to, subject, text_body, html_body=None):
        """Send one email synchronously. Returns True on success."""
        if not self.enabled:
            logger.info(f"Email not configured (SMTP_HOST/MAIL_FROM unset); skipping '{subject}' to {to}")
            return False

        try:
            msg = EmailMessage()
            msg['Subject'] = subject
            msg['From'] = self.sender
            msg['To'] = to  # EmailMessage rejects header injection (newlines) here
            msg.set_content(text_body)
            if html_body:
                msg.add_alternative(html_body, subtype='html')

            context = ssl.create_default_context()
            if self.port == 465:
                with smtplib.SMTP_SSL(self.host, self.port, context=context, timeout=15) as server:
                    self._login_and_send(server, msg)
            else:
                with smtplib.SMTP(self.host, self.port, timeout=15) as server:
                    server.starttls(context=context)
                    self._login_and_send(server, msg)
            logger.info(f"Sent '{subject}' to {to}")
            return True
        except Exception as e:
            logger.error(f"Failed to send '{subject}' to {to}: {str(e)}")
            return False

    def _login_and_send(self, server, msg):
        if self.username and self.password:
            server.login(self.username, self.password)
        server.send_message(msg)

    def send_async(self, *args, **kwargs):
        """Send in a background thread so the HTTP request never waits on SMTP."""
        threading.Thread(target=self.send, args=args, kwargs=kwargs, daemon=True).start()

    def send_welcome_email(self, to, username):
        """Welcome email sent after successful registration (non-blocking)."""
        subject = "Welcome to Product Recommendations!"
        link_text = f"\n\nStart exploring: {self.app_url}" if self.app_url else ""
        text_body = (
            f"Hi {username},\n\n"
            "Thanks for creating an account! Set your favorite categories, brands and "
            "price range, browse a few products, and we'll start tailoring "
            f"recommendations to you.{link_text}\n\n"
            "Happy shopping!"
        )

        safe_name = html.escape(username)
        link_html = (
            f'<p><a href="{html.escape(self.app_url, quote=True)}" '
            'style="background:#4f46e5;color:#fff;padding:10px 18px;border-radius:6px;'
            'text-decoration:none;display:inline-block">Start exploring</a></p>'
            if self.app_url else ""
        )
        html_body = f"""\
<html>
  <body style="font-family:Arial,Helvetica,sans-serif;color:#222;line-height:1.5">
    <h2>Welcome, {safe_name}!</h2>
    <p>Thanks for creating an account. Set your favorite categories, brands and
       price range, browse a few products, and we'll start tailoring
       recommendations to you.</p>
    {link_html}
    <p>Happy shopping!</p>
  </body>
</html>
"""
        self.send_async(to, subject, text_body, html_body)
