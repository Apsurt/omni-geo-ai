# CLAUDE.md - Assistant Guidelines

## Build, Test & Run Commands
- Run scraper: `python scripts/fetch_panoramas.py --num_images <number>`
- Train model: `python train.py --model vit_hemisphere --epochs <number>`
- Database backup: `python scripts/backup_db.py`
- Single test: `pytest tests/<test_file>.py::test_<function_name>`
- Lint: `flake8 . --count --exit-zero --max-complexity=10 --max-line-length=127 --ignore=E203,W503`
- Type check: `mypy --ignore-missing-imports .`

## Code Standards & Conventions
- **Python version**: 3.10+
- **Formatting**: Black with 127 character line limit
- **Imports**: Group standard lib, third-party, local (alphabetical within groups)
- **Naming**: snake_case for variables/functions, PascalCase for classes
- **Types**: Use type hints throughout (leverage mypy)
- **Documentation**: Google-style docstrings with Parameters/Returns
- **Error handling**: Use explicit exception types with context messages
- **Database**: Use SQLAlchemy models with explicit schemas
- **API keys**: Never hardcode, use .env files with python-dotenv
- **Data Pipeline**: Validate data exists before processing
- **Tensor ops**: Prefer PyTorch's built-in ops over manual implementations