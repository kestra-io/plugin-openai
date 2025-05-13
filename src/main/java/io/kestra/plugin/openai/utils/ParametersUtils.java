package io.kestra.plugin.openai.utils;

import com.fasterxml.jackson.core.type.TypeReference;
import com.fasterxml.jackson.databind.ObjectMapper;
import com.openai.core.JsonValue;
import com.openai.models.responses.*;
import io.kestra.core.exceptions.IllegalVariableEvaluationException;
import io.kestra.core.models.property.Property;
import io.kestra.core.runners.RunContext;
import io.kestra.core.serializers.JacksonMapper;

import java.io.IOException;
import java.io.InputStream;
import java.net.URI;
import java.net.URLConnection;
import java.util.*;
import java.util.stream.Collectors;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import static io.kestra.core.utils.Rethrow.throwFunction;

public final class ParametersUtils {
    private static final Logger logger = LoggerFactory.getLogger(ParametersUtils.class);
    public static final ObjectMapper OBJECT_MAPPER = JacksonMapper.ofJson();
    private static final String BASE64_PREFIX = "data:";

    private ParametersUtils() {
        // Prevent instantiation of utility class
    }

    public static List<ResponseInputItem.Message> listParameters(RunContext runContext, Property<Object> parameters) throws Exception {
        Object rendered = runContext.render(parameters).as(Object.class).orElseThrow();
        String renderedString = (String) rendered;
        String trimmedString = renderedString.trim();

        if (trimmedString.startsWith("[") && trimmedString.endsWith("]")) {
            List<Map<String, Object>> rawList = OBJECT_MAPPER.readValue(renderedString, new TypeReference<>() {
            });
            return convertToMessages(runContext, rawList);
        }

        return List.of(ResponseInputItem.Message.builder()
            .role(ResponseInputItem.Message.Role.USER)
            .addContent(ResponseInputContent.ofInputText(
                ResponseInputText.builder()
                    .text(renderedString)
                    .build()
            ))
            .build());
    }

    private static List<ResponseInputItem.Message> convertToMessages(RunContext runContext, List<Map<String, Object>> renderedList) throws Exception {
        List<ResponseInputItem.Message> messages = OBJECT_MAPPER.convertValue(
            renderedList, new TypeReference<>() {
            }
        );

        return messages.stream()
            .map(throwFunction(message ->
                ResponseInputItem.Message.builder()
                    .role(message.role())
                    .content(
                        message.content().stream()
                            .map(throwFunction(content -> {
                                if (content.inputImage().isPresent()) {
                                    return processImageContent(runContext, content);
                                } else if (content.inputText().isPresent()) {
                                    return processTextContent(content);
                                } else if (content.inputFile().isPresent()) {
                                    return processFileContent(content);
                                } else {
                                    return content;
                                }
                            }))
                            .collect(Collectors.toList())
                    )
                    .build()
            ))
            .collect(Collectors.toList());
    }

    private static ResponseInputContent processFileContent(ResponseInputContent content) {
        ResponseInputFile file = content.asInputFile();
        ResponseInputFile.Builder builder = ResponseInputFile.builder();

        file.fileId().ifPresent(builder::fileId);
        file.fileData().ifPresent(builder::fileData);
        file.filename().ifPresent(builder::filename);

        return ResponseInputContent.ofInputFile(builder.build());
    }

    /**
     * Processes image content, converting Kestra URLs to base64 if necessary.
     *
     * @param runContext The current run context
     * @param content The input content to process
     * @return Processed image content
     * @throws IOException if there's an error reading the image
     */
    private static ResponseInputContent processImageContent(RunContext runContext, ResponseInputContent content) throws IOException, IllegalVariableEvaluationException {
        ResponseInputImage image = content.asInputImage();

        if (image.imageUrl().isPresent()) {
            String renderedUrl = runContext.render(image.imageUrl().get());

            String processedUrl;
            if (renderedUrl.startsWith("kestra:///")) {
                if (image._additionalProperties().isEmpty() || image._additionalProperties().get("mimeType").isMissing())
                   throw new IllegalArgumentException("You must provide 'mimeType' as an additional parameter, when using a file from 'input: FILE'");
                }

            processedUrl = convertKestraUrlToBase64(runContext, renderedUrl,image._additionalProperties().get("mimeType"));

            ResponseInputImage.Detail detail;
            try {
                detail = image.detail() != null ? image.detail() : ResponseInputImage.Detail.AUTO;
            } catch (Exception e) {
                logger.debug("Error getting image detail, defaulting to AUTO: {}", e.getMessage());
                detail = ResponseInputImage.Detail.AUTO;
            }

            return ResponseInputContent.ofInputImage(
                ResponseInputImage.builder()
                    .imageUrl(processedUrl)
                    .detail(detail)
                    .build());
        }
        return content;
    }

    private static ResponseInputContent processTextContent(ResponseInputContent content) {
        ResponseInputText text = content.asInputText();
        return ResponseInputContent.ofInputText(
            ResponseInputText.builder()
                .text(text.text())
                .build()
        );
    }

    /**
     * Converts a Kestra storage URL to a base64 data URI.
     *
     * @param runContext The current run context
     * @param kestraUrl The Kestra storage URL to convert
     * @return Base64 encoded data URI
     * @throws IOException if there's an error reading the file
     */
    private static String convertKestraUrlToBase64(RunContext runContext, String kestraUrl, JsonValue mimeType) throws IOException {
        try (InputStream in = runContext.storage().getFile(URI.create(kestraUrl))) {
            byte[] bytes = in.readAllBytes();
            String encoded = Base64.getEncoder().encodeToString(bytes);

            String filename = kestraUrl.substring(kestraUrl.lastIndexOf('/') + 1);
            String renderedMimeType = mimeType.isNull() ? URLConnection.guessContentTypeFromName(filename) : mimeType.toString();

            return BASE64_PREFIX + renderedMimeType + ";base64," + encoded;
        }
    }

    /**
     * Extracts output text from a list of ResponseOutputItem.
     *
     * @param items List of output items
     * @return Extracted text from the output items
     */
    public static String extractOutputText(List<ResponseOutputItem> items) {
        return items.stream()
            .map(ParametersUtils::extractFromItem)
            .filter(Objects::nonNull)
            .collect(Collectors.joining("\n"));
    }

    /**
     * Extracts text from a single ResponseOutputItem.
     *
     * @param item The output item to extract text from
     * @return Extracted text or null
     */
    private static String extractFromItem(ResponseOutputItem item) {
        if (item == null) return null;

        if (item.functionCall().isPresent()) {
            return item.functionCall().get().arguments();
        }

        return item.message().stream()
            .flatMap(message -> message.content().stream())
            .flatMap(content -> content.outputText().stream())
            .map(ResponseOutputText::text)
            .collect(Collectors.joining("\n"));
    }
}