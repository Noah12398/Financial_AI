# Financial AI 🤖💰

A comprehensive personal finance management application powered by artificial intelligence to help users track expenses, manage budgets, and make informed financial decisions.

## Features

### 🔐 User Management
- **User Registration**: Create new accounts with secure authentication
- **User Login**: Secure login system for existing users
- **Dashboard**: Comprehensive overview of your financial status

### 💳 Transaction Management
- **View Transactions**: Display all financial transactions in an organized manner
- **Add Transactions**: Record new income and expense transactions
- **Edit Transactions**: Modify existing transaction details
- **Transaction Categories**: Organize transactions by customizable categories

### 📊 Budget Management
- **Budget Overview**: Monitor your budget performance and remaining allocations
- **Create Budgets**: Set up budgets for different categories and time periods
- **Edit Budgets**: Adjust budget limits and parameters as needed

### 💸 Expense Tracking
- **Expense Dashboard**: Visualize spending patterns and trends
- **Add Expenses**: Quick expense entry with category assignment
- **Edit Expenses**: Modify expense details and categorization
- **Expense Analytics**: AI-powered insights into spending behavior

### 🏷️ Category Management
- **Custom Categories**: Create and manage personalized expense and income categories
- **Category Analytics**: Track spending across different categories
- **Smart Categorization**: AI-assisted automatic transaction categorization

### 🤖 AI-Powered Chatbot
- **Financial Assistant**: Interactive chatbot for financial queries and advice
- **Smart Insights**: AI-generated financial recommendations and tips
- **Natural Language Processing**: Ask questions about your finances in plain English
- **Personalized Guidance**: Tailored financial advice based on your spending patterns

## Technology Stack

- **Frontend**: Modern web technologies for responsive user interface
- **Backend**: Robust server architecture for data processing
- **AI Integration**: Machine learning models for financial insights
- **Database**: Secure data storage for user information and transactions
- **Authentication**: Secure user authentication and session management

## Getting Started

### Prerequisites
- Django
- Database system (PostgreSQL recommended)
- API keys for AI services (if applicable)

### Installation

1. Clone the repository:
```bash
git clone https://github.com/Noah12398/Financial_AI.git
cd Financial_AI
```

2. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Set up environment variables:
```bash
cp .env.example .env
# Edit .env with your configuration
```

5. Initialize the database:
```bash
python manage.py makemigrations
python manage.py migrate
```

6. Create a superuser (optional):
```bash
python manage.py createsuperuser
```

7. Start the development server:
```bash
python manage.py runserver
```

## Usage

1. **Registration**: Create a new account or log in with existing credentials
2. **Dashboard Setup**: Complete your profile and financial goals
3. **Add Transactions**: Start by importing or manually adding your transactions
4. **Create Budgets**: Set up budgets for different spending categories
5. **Track Expenses**: Monitor your spending and receive AI-powered insights
6. **Chat with AI**: Use the chatbot for financial advice and questions

## Key Features in Detail

### AI-Powered Insights
- Spending pattern analysis
- Budget optimization recommendations
- Unusual transaction detection
- Savings opportunity identification
- Financial goal tracking

### Smart Categorization
- Automatic transaction categorization using machine learning
- Custom category creation and management
- Spending trend analysis by category

### Interactive Dashboard
- Real-time financial overview
- Visual charts and graphs
- Budget vs. actual spending comparisons
- Monthly and yearly financial summaries

## Contributing

We welcome contributions to improve Financial AI! Please follow these steps:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## Screenshots

User Registration Screen

![Screenshot 2025-03-07 013835](https://github.com/user-attachments/assets/5cc144f7-23fd-482c-983c-f8334f2f12b5)

User Login Screen

![Screenshot 2025-03-07 013820](https://github.com/user-attachments/assets/c98056bc-785f-440e-8263-681f86099c9e)

Dashboard

![Screenshot 2025-03-07 013410](https://github.com/user-attachments/assets/c4b3bea1-3a95-42ed-8268-523de0324272)

Transactions Screen

![Screenshot 2025-03-07 013427](https://github.com/user-attachments/assets/f354d37d-c679-4351-8646-b34edb55920e)

Add Transactions Screen

![Screenshot 2025-03-07 013459](https://github.com/user-attachments/assets/077b3cf8-c108-4710-84fc-02c3c56dd599)

Edit Transactions Screen

![Screenshot 2025-03-07 013443](https://github.com/user-attachments/assets/6b04c0c5-3073-4f28-9a26-29cd72785aa2)

Budget Screen

![Screenshot 2025-03-07 013524](https://github.com/user-attachments/assets/7a20d53b-219c-4fcd-86a3-b07f3ae9b2f3)

Add Budget Screen

![Screenshot 2025-03-07 013624](https://github.com/user-attachments/assets/aa018eff-635e-4f67-9319-1039f4982a73)

Edit Budget Screen

![Screenshot 2025-03-07 013534](https://github.com/user-attachments/assets/78d87c79-423a-4854-9e41-1aa92fb739fd)

Expense Screen

![Screenshot 2025-03-07 013648](https://github.com/user-attachments/assets/ffcde6b6-801c-43cd-b628-2b3b50a98066)

Add Expense Screen

![Screenshot 2025-03-07 013713](https://github.com/user-attachments/assets/8a17ad71-1bad-465a-a9e7-e74887f85c3b)

Edit Expense Screen

![Screenshot 2025-03-07 013659](https://github.com/user-attachments/assets/98fe77dd-bd2f-4dbc-91f3-e4ef8d5b5814)

Category Screen

![Screenshot 2025-03-07 013729](https://github.com/user-attachments/assets/60b16e37-355b-47ff-bfca-43a1545f8437)

Chatbot Screen

![Screenshot 2025-03-07 013807](https://github.com/user-attachments/assets/8216d782-4bc1-485c-b516-b2aa4ca4b737)

