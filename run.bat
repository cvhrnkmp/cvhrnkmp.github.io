set CGO_ENABLED=1
go run -race .
go install -tags extended github.com/gohugoio/hugo@latest