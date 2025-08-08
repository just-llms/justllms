set -e  # Exit on error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    echo -e "${GREEN}[✓]${NC} $1"
}

print_error() {
    echo -e "${RED}[✗]${NC} $1"
    exit 1
}

print_warning() {
    echo -e "${YELLOW}[!]${NC} $1"
}

# Check if version argument is provided
if [ -z "$1" ]; then
    echo "Usage: ./release.sh <version> <release_type> [description]"
    echo "Release types: patch, minor, major"
    echo "Example: ./release.sh 1.0.2 patch"
    exit 1
fi

VERSION=$1
RELEASE_TYPE=${2:-patch}
DESCRIPTION=${3:-""}

# Validate version format (x.y.z)
if ! [[ $VERSION =~ ^[0-9]+\.[0-9]+\.[0-9]+$ ]]; then
    print_error "Invalid version format. Use x.y.z (e.g., 1.0.2)"
fi

echo "========================================="
echo "    JustLLMs Release Script v1.0"
echo "========================================="
echo "Version: $VERSION"
echo "Type: $RELEASE_TYPE"
echo "Description: $DESCRIPTION"
echo "========================================="
echo ""

# Check for uncommitted changes
if ! git diff-index --quiet HEAD --; then
    print_warning "You have uncommitted changes:"
    git status --short
    read -p "Do you want to commit them first? (y/n): " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        git add -A
        read -p "Enter commit message: " commit_msg
        git commit -m "$commit_msg"
    else
        print_error "Please commit or stash your changes before releasing"
    fi
fi

# Pull latest changes
print_status "Pulling latest changes from main..."
git checkout main
git pull origin main

# Update version in files
print_status "Updating version to $VERSION..."

# Update pyproject.toml
if [[ "$OSTYPE" == "darwin"* ]]; then
    # macOS
    sed -i '' "s/version = \".*\"/version = \"$VERSION\"/" pyproject.toml
else
    # Linux
    sed -i "s/version = \".*\"/version = \"$VERSION\"/" pyproject.toml
fi

# Update __version__.py
if [[ "$OSTYPE" == "darwin"* ]]; then
    sed -i '' "s/__version__ = \".*\"/__version__ = \"$VERSION\"/" justllms/__version__.py
else
    sed -i "s/__version__ = \".*\"/__version__ = \"$VERSION\"/" justllms/__version__.py
fi

# Update CHANGELOG.md
print_status "Updating CHANGELOG.md..."
TODAY=$(date +%Y-%m-%d)

# Prepare changelog entry based on release type
case $RELEASE_TYPE in
    patch)
        CHANGELOG_ENTRY="## [$VERSION] - $TODAY\n\n### Fixed\n- $DESCRIPTION\n\n"
        ;;
    minor)
        CHANGELOG_ENTRY="## [$VERSION] - $TODAY\n\n### Added\n- $DESCRIPTION\n\n"
        ;;
    major)
        CHANGELOG_ENTRY="## [$VERSION] - $TODAY\n\n### Changed\n- $DESCRIPTION\n\n### Breaking Changes\n- \n\n"
        ;;
    *)
        CHANGELOG_ENTRY="## [$VERSION] - $TODAY\n\n### Changed\n- $DESCRIPTION\n\n"
        ;;
esac

# Add changelog entry after the header
if [[ "$OSTYPE" == "darwin"* ]]; then
    sed -i '' "/# Changelog/a\\
$CHANGELOG_ENTRY" CHANGELOG.md
else
    sed -i "/# Changelog/a\\$CHANGELOG_ENTRY" CHANGELOG.md
fi

# Show the changes
print_status "Version updated in:"
echo "  - pyproject.toml"
echo "  - justllms/__version__.py"
echo "  - CHANGELOG.md"

# Commit version bump
print_status "Committing version bump..."
git add pyproject.toml justllms/__version__.py CHANGELOG.md
git commit -m "Release v$VERSION

Type: $RELEASE_TYPE
${DESCRIPTION:+Description: $DESCRIPTION}"

# Create git tag
print_status "Creating git tag v$VERSION..."
if [ -z "$DESCRIPTION" ]; then
    git tag -a "v$VERSION" -m "Release version $VERSION"
else
    git tag -a "v$VERSION" -m "Release version $VERSION - $DESCRIPTION"
fi

# Push changes and tag
print_status "Pushing to GitHub..."
git push origin main
git push origin "v$VERSION"

# Clean build artifacts
print_status "Cleaning build artifacts..."
rm -rf dist/ build/ *.egg-info/ __pycache__/

# Build distribution
print_status "Building distribution packages..."
python -m build

# Check build output
if [ ! -f "dist/justllms-$VERSION.tar.gz" ] || [ ! -f "dist/justllms-$VERSION-py3-none-any.whl" ]; then
    print_error "Build failed! Expected files not found in dist/"
fi

print_status "Build successful! Found:"
ls -la dist/

# Upload to PyPI
print_status "Uploading to PyPI..."
read -p "Do you want to upload to TestPyPI first? (y/n): " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    print_status "Uploading to TestPyPI..."
    python -m twine upload --repository testpypi dist/*
    echo ""
    print_status "Test installation with:"
    echo "  pip install --index-url https://test.pypi.org/simple/ justllms==$VERSION"
    echo ""
    read -p "Continue with PyPI upload? (y/n): " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        print_warning "Skipping PyPI upload. You can manually upload later with:"
        echo "  python -m twine upload dist/*"
        exit 0
    fi
fi

python -m twine upload dist/*

# Create GitHub release
print_status "Creating GitHub release..."
echo ""
echo "Release notes for v$VERSION:"
echo "================================="
echo "$CHANGELOG_ENTRY"
echo "================================="
echo ""
read -p "Create GitHub release with gh CLI? (y/n): " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    # Check if gh is installed
    if command -v gh &> /dev/null; then
        gh release create "v$VERSION" \
            --title "v$VERSION" \
            --notes "$CHANGELOG_ENTRY" \
            --latest
        print_status "GitHub release created!"
    else
        print_warning "GitHub CLI (gh) not installed. Create release manually at:"
        echo "  https://github.com/just-llms/justllms/releases/new?tag=v$VERSION"
    fi
else
    print_warning "Remember to create GitHub release at:"
    echo "  https://github.com/just-llms/justllms/releases/new?tag=v$VERSION"
fi

# Final verification
print_status "Verifying PyPI release..."
sleep 5  # Give PyPI a moment to update
pip install --upgrade justllms==$VERSION --dry-run

echo ""
echo "========================================="
echo -e "${GREEN}    ✨ Release v$VERSION Complete! ✨${NC}"
echo "========================================="
echo ""
echo "Next steps:"
echo "  1. Verify installation: pip install --upgrade justllms"
echo "  2. Test import: python -c 'import justllms; print(justllms.__version__)'"
echo "  3. Check PyPI: https://pypi.org/project/justllms/"
echo "  4. Check GitHub: https://github.com/just-llms/justllms/releases"
echo ""
echo "To announce the release:"
echo "  - Update project documentation"
echo "  - Post on social media"
echo "  - Notify users via GitHub discussions"
echo ""