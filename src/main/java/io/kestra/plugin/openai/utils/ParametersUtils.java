package io.kestra.plugin.openai.utils;

import com.fasterxml.jackson.core.type.TypeReference;
import com.fasterxml.jackson.databind.ObjectMapper;
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

import static io.kestra.core.utils.Rethrow.throwFunction;

public final class ParametersUtils {
    public static final ObjectMapper OBJECT_MAPPER = JacksonMapper.ofJson();

    private ParametersUtils() {
        // prevent instantiation
    }

    public static List<ResponseInputItem.Message> listParameters(RunContext runContext, Property<Object> parameters) throws Exception {

        Object rendered = runContext.render(parameters).as(Object.class).orElseThrow();

        String renderedString = (String) rendered;

        String trimmedString = renderedString.trim();
        if (trimmedString.startsWith("[") && trimmedString.endsWith("]")) {
            List<Map<String, Object>> rawList = OBJECT_MAPPER.readValue(renderedString, new TypeReference<>() {});
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
            renderedList, new TypeReference<>() {}
        );

        return messages.stream()
            .map(throwFunction(message -> {
                return ResponseInputItem.Message.builder()
                    .role(message.role())
                    .content(
                        message.content().stream()
                            .map(throwFunction(content -> {
                                if (content.inputImage().isPresent()) {
                                    ResponseInputImage image = content.asInputImage();

                                    // Check if we have a URL
                                    if (image.imageUrl().isPresent()) {
                                        String renderedUrl = runContext.render(image.imageUrl().get());

                                        String processedUrl;
                                        if (renderedUrl.startsWith("kestra:///")) {
                                            processedUrl = convertKestraUrlToBase64(runContext, renderedUrl);
                                        } else {
                                            processedUrl = renderedUrl;
                                        }

                                        return ResponseInputContent.ofInputImage(
                                            ResponseInputImage.builder()
                                                .imageUrl(processedUrl)
                                                .detail(ResponseInputImage.Detail.AUTO)
                                                .build()
                                        );
                                    } else {
                                        // Check for file ID
                                        if (image.fileId().isPresent()) {
                                            runContext.logger().debug("Input image file ID: {}", image.fileId().get());
                                            return ResponseInputContent.ofInputImage(
                                                ResponseInputImage.builder()
                                                    .fileId(image.fileId().get())
                                                    .detail(ResponseInputImage.Detail.AUTO) // Always set a non-null value
                                                    .build()
                                            );
                                        } else {
                                            runContext.logger().warn("Image without URL or fileId, skipping");
                                            return content; // Return original content as fallback
                                        }
                                    }
                                } else if (content.inputFile().isPresent()) {
                                    ResponseInputFile file = content.asInputFile();

                                    // Check for file ID first (preferred)
                                    if (file.fileId().isPresent()) {
                                        return ResponseInputContent.ofInputFile(
                                            ResponseInputFile.builder()
                                                .fileId(file.fileId().get())
                                                .build()
                                        );
                                    }
                                    // Fall back to file data and filename if available
                                    else if (file.fileData().isPresent() && file.filename().isPresent()) {
                                        return ResponseInputContent.ofInputFile(
                                            ResponseInputFile.builder()
                                                .fileData(file.fileData().get())
                                                .filename(file.filename().get())
                                                .build()
                                        );
                                    } else {
                                        runContext.logger().warn("File without fileId or fileData+filename, skipping");
                                        return content; // Return original content as fallback
                                    }
                                } else if (content.inputText().isPresent()) {
                                    ResponseInputText text = content.asInputText();
                                    return ResponseInputContent.ofInputText(
                                        ResponseInputText.builder()
                                            .text(text.text())
                                            .build()
                                    );
                                } else {
                                    return content;
                                }
                            }))
                            .collect(Collectors.toList())
                    )
                    .build();
            }))
            .collect(Collectors.toList());
    }

    /**
     * Converts a Kestra storage URL to a base64 data URI
     */
    private static String convertKestraUrlToBase64(RunContext runContext, String kestraUrl) throws IOException {
        try (InputStream in = runContext.storage().getFile(URI.create(kestraUrl))) {
            byte[] bytes = in.readAllBytes();
            String encoded = Base64.getEncoder().encodeToString(bytes);

            String filename = kestraUrl.substring(kestraUrl.lastIndexOf('/') + 1);
            String mimeType = URLConnection.guessContentTypeFromName(filename);
            if (mimeType == null) {
                if (filename.toLowerCase().endsWith(".jpg") || filename.toLowerCase().endsWith(".jpeg")) {
                    mimeType = "image/jpeg";
                } else if (filename.toLowerCase().endsWith(".png")) {
                    mimeType = "image/png";
                } else {
                    mimeType = "application/octet-stream";
                }
            }

            return "data:" + mimeType + ";base64," + encoded;
        }
    }

    public static String extractOutputText(List<ResponseOutputItem> items) {
        return items.stream()
            .map(ParametersUtils::extractFromItem)
            .filter(Objects::nonNull)
            .collect(Collectors.joining("\n"));
    }

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