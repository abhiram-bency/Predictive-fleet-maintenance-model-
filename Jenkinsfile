pipeline {
    agent any

    stages {

        stage('Build Docker Image') {
            steps {
                bat 'docker build -t fleet-maintenance-app .'
            }
        }

        stage('Stop Existing Container') {
            steps {
                bat 'docker stop fleet-maintenance-container || exit 0'
                bat 'docker rm fleet-maintenance-container || exit 0'
            }
        }

        stage('Run New Container') {
            steps {
                bat 'docker run -d -p 8501:8501 --name fleet-maintenance-container fleet-maintenance-app'
            }
        }
    }
}