.PHONY: format build run stop clean

# Formatting commands
format:
	isort --force-single-line-imports .
	autoflake --remove-all-unused-imports --recursive --remove-unused-variables --in-place . --exclude=__init__.py
	black .
	isort --recursive --apply .

# Docker commands
build:
	docker build -t telegram-bot .

run:
	docker run --env-file .env -p 5000:5000 --name telegram-bot telegram-bot

stop:
	docker stop telegram-bot
	docker rm telegram-bot

clean:
	docker rmi telegram-bot

# Default goal
.DEFAULT_GOAL := help

# Help target to list available commands
help:
	@echo "Available commands:"
	@echo "  format - Run code formatting tools"
	@echo "  build  - Build the Docker image"
	@echo "  run    - Run the Docker container"
	@echo "  stop   - Stop and remove the Docker container"
	@echo "  clean  - Remove the Docker image"
