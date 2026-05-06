name: Pull Request
description: Submit a Pull Request to improve the TVRA project
title: '[PR] '
labels: ['pull-request']
body:
  - type: markdown
    attributes:
      value: |
        ## Pull Request Description

        Thank you for contributing to the TVRA project! Please fill out the information below to help us review your PR more quickly.

  - type: textarea
    id: description
    attributes:
      label: PR Description
      description: Please describe in detail what this PR changes and why these changes are needed
      placeholder: |
        ## Changes
        - 

        ## Reasons for Changes
        - 

        ## Impact Scope
        - 
    validations:
      required: true

  - type: dropdown
    id: type
    attributes:
      label: PR Type
      options:
        - Please select...
        - 🚀 New Feature
        - 🐛 Bug Fix
        - 📚 Documentation
        - ♻️ Refactoring
        - ⚡ Performance
        - 🧪 Testing
        - 🔧 Tooling
        - Other
    validations:
      required: true

  - type: textarea
    id: related-issue
    attributes:
      label: Related Issue
      description: Please enter the related Issue number (e.g., #123)
      placeholder: '#'

  - type: textarea
    id: testing
    attributes:
      label: Testing Instructions
      description: Please explain how to test these changes
      placeholder: |
        ### Testing Steps
        1. 
        2. 
        3. 

        ### Expected Results
        - 
  - type: textarea
    id: checklist
    attributes:
      label: Checklist
      description: Please confirm the following items before submitting
      placeholder: |
        - [ ] My code follows the existing code style
        - [ ] I have tested these changes
        - [ ] I have updated the relevant documentation (if necessary)
        - [ ] I have clearly described the changes in the commit message

  - type: textarea
    id: screenshots
    attributes:
      label: Screenshots / Test Results (if any)
      description: If there are UI changes or test results, please attach screenshots
      placeholder: |
        ### Before
        ![Before](url)

        ### After
        ![After](url)
